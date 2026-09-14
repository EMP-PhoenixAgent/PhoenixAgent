//! The state capsule — the exe carries its OWN private state, encrypted.
//!
//! What belongs inside: everything that is NOT rebuildable and NOT public —
//! the encrypted memory DB, config, the key bundle, the 2FA flag, logs.
//! What does NOT: **models** (public artifacts, GiB-scale — they live in a
//! plain `models/` folder BESIDE the exe, managed by the normal pull flow,
//! never sealed, never re-copied) and the WebView2 cache (machine-local).
//! The capsule is therefore small (MBs), and every seal is a cheap full
//! rewrite — no append/carry machinery.
//!
//! v1 on-disk layout (legacy, kept readable so pre-Phase-C exes migrate at
//! their first unlock):
//!
//! ```text
//! [ exe code region  (immutable, never rewritten) ]
//! [ zip body         (standard PKZIP, PLAINTEXT)  ]
//! [ b"PHXCAPS1"      (8-byte marker) ]
//! [ u64 LE           (zip body length) ]
//! ```
//!
//! v2 layout — the sealed zone. Every entry's payload is ciphertext (chunked
//! AES-256-GCM), every entry NAME is ciphertext, and a small plaintext tail
//! header carries the wrapped capsule key (a serialized `KeyBundle`, opaque
//! here) so the app can gate and decrypt before any state exists on disk:
//!
//! ```text
//! [ exe code region  (immutable, never rewritten) ]
//! [ zip body         (names + payloads encrypted; structure is zip) ]
//! [ b"PHXCAPS2"      (8-byte marker) ]
//! [ u64 LE           (zip body length) ]
//! [ header bytes     (wrapped-key bundle; 8..=8192 bytes, opaque) ]
//! [ u16 LE           (header length) ]
//! ```
//!
//! The tail is self-describing: [`Capsule::open`] / [`Capsule::probe`] read
//! from EOF and, when a marker matches, know exactly where the capsule
//! begins — the exe code region ends there. Trailing bytes after a PE image
//! are ignored by the OS loader, so the artifact stays a valid exe at any
//! size. Without the capsule key a v2 tail is opaque: entry names fail
//! authentication and no payload can be read.
//!
//! Per-entry encryption format (v2 payloads):
//!
//! ```text
//! [ u64 entry nonce prefix ][ chunk0 ct+tag ][ chunk1 ct+tag ] ...
//! ```
//!
//! Plaintext is chunked at [`CHUNK_PT`] (1 MiB); each chunk is sealed with
//! AES-256-GCM under the capsule key with nonce `[entry prefix (8) | chunk
//! index u32 LE (4)]` — chunks authenticate independently, so entries of any
//! size stream in constant memory and a tampered byte anywhere fails the
//! tag check of exactly its chunk. The zip `crc32` is 0 for v2 entries (GCM
//! is the integrity check); `comp_size` counts the ciphertext total
//! (prefix + chunks incl. tags) and `uncomp_size` the plaintext length.

use std::fs;
use std::io::{self, Read, Seek, Write};
use std::path::{Path, PathBuf};

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use rand::RngCore;
use zeroize::Zeroize;

const MARKER_V1: &[u8; 8] = b"PHXCAPS1";
const MARKER_V2: &[u8; 8] = b"PHXCAPS2";
/// Stream chunk for large entries.
const CHUNK: usize = 8 * 1024 * 1024;
/// Plaintext chunk size for v2 entry encryption.
const CHUNK_PT: usize = 1024 * 1024;
/// AES-GCM tag length.
const TAG: usize = 16;
/// v2 tail header bounds (the serialized key bundle).
const HEADER_MIN: usize = 8;
const HEADER_MAX: usize = 8192;

/// The 32-byte key encrypting a v2 (sealed) capsule. Random entropy from
/// [`crate::crypto::random_key`]; stored only inside the key bundle's wraps.
pub type CapsuleKey = [u8; 32];

const SIG_LOCAL: u32 = 0x0403_4b50;
const SIG_CD: u32 = 0x0201_4b50;
const SIG_EOCD: u32 = 0x0605_4b50;

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn u64le(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

fn corrupt(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("capsule corrupt: {msg}"))
}

/// Error text when a sealed capsule is opened without its key.
const SEALEDMSG: &str = "capsule is sealed — the key is required to open it";

fn ceil_div(a: u64, b: u64) -> u64 {
    if a == 0 {
        0
    } else {
        (a + b - 1) / b
    }
}

// ===== v2 crypto helpers =======================================================

fn gcm(key: &CapsuleKey) -> Aes256Gcm {
    Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key))
}

/// Chunk nonce: `[entry prefix (8) | chunk index u32 LE (4)]`.
fn chunk_nonce(prefix: &[u8; 8], idx: u32) -> [u8; 12] {
    let mut n = [0u8; 12];
    n[..8].copy_from_slice(prefix);
    n[8..].copy_from_slice(&idx.to_le_bytes());
    n
}

/// Encrypt one entry NAME: `[nonce (12) || ct+tag]`. Random nonce per name —
/// names are few and only ever matched after full decryption, so randomness
/// (not determinism) is the right tool.
fn enc_name(key: &CapsuleKey, plain: &[u8]) -> io::Result<Vec<u8>> {
    let mut nonce = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce);
    let ct = gcm(key)
        .encrypt(Nonce::from_slice(&nonce), plain)
        .map_err(|_| corrupt("name encrypt"))?;
    let mut out = Vec::with_capacity(12 + ct.len());
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ct);
    Ok(out)
}

fn dec_name(key: &CapsuleKey, raw: &[u8]) -> io::Result<String> {
    if raw.len() < 12 + TAG {
        return Err(corrupt("encrypted entry name too short"));
    }
    let (nonce, ct) = raw.split_at(12);
    let pt = gcm(key)
        .decrypt(Nonce::from_slice(nonce), ct)
        .map_err(|_| corrupt("entry name failed authentication (wrong key or tampered exe)"))?;
    String::from_utf8(pt).map_err(|_| corrupt("entry name is not valid UTF-8"))
}

/// Encrypt a complete (in-memory) payload: `prefix (8) || chunks`.
fn enc_payload(key: &CapsuleKey, plain: &[u8]) -> io::Result<Vec<u8>> {
    let mut prefix = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut prefix);
    let cipher = gcm(key);
    let nchunks = ceil_div(plain.len() as u64, CHUNK_PT as u64) as usize;
    let mut out = Vec::with_capacity(8 + plain.len() + TAG * nchunks);
    out.extend_from_slice(&prefix);
    for i in 0..nchunks {
        let pt = &plain[i * CHUNK_PT..plain.len().min((i + 1) * CHUNK_PT)];
        let ct = cipher
            .encrypt(Nonce::from_slice(&chunk_nonce(&prefix, i as u32)), pt)
            .map_err(|_| corrupt("payload encrypt"))?;
        out.extend_from_slice(&ct);
    }
    Ok(out)
}

/// Ciphertext framing implied by [`enc_payload`] for a known plaintext length.
fn sealed_comp_size(pt_len: u64) -> u64 {
    8 + pt_len + TAG as u64 * ceil_div(pt_len, CHUNK_PT as u64)
}

// ===== read side ===============================================================

/// Which capsule generation an exe carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapsuleVer {
    /// Plaintext zip tail (pre-Phase-C).
    V1,
    /// Encrypted sealed zone.
    V2,
}

#[derive(Debug, Clone)]
pub struct CapsuleEntry {
    /// Decrypted (v2) / plaintext (v1) virtual path.
    pub name: String,
    /// Name bytes exactly as stored in the zip (encrypted for v2) — the
    /// central directory and local headers carry these verbatim on re-seal.
    pub name_raw: Vec<u8>,
    /// 0 = stored, 8 = deflate (of the plaintext).
    pub method: u16,
    /// 0 for v2 entries (GCM is the integrity check); real CRC for v1.
    pub crc32: u32,
    /// Ciphertext total for v2 (incl. nonce prefix + tags); stored length for v1.
    pub comp_size: u64,
    /// Plaintext (original file) length.
    pub uncomp_size: u64,
    /// Local-header offset, relative to the zip body start.
    pub local_offset: u64,
}

enum Tail {
    V1 { zip_len: u64 },
    V2 { zip_len: u64, header: Vec<u8> },
}

/// Read the self-describing tail. `Ok(None)` = no capsule (fresh build).
fn read_tail(f: &mut fs::File, len: u64) -> io::Result<Option<Tail>> {
    if len < 22 + 16 + 2 {
        return Ok(None);
    }
    // v2 candidate: the file ends with `u16 header_len`.
    let mut b2 = [0u8; 2];
    f.seek(io::SeekFrom::Start(len - 2))?;
    f.read_exact(&mut b2)?;
    let hl = u16le(&b2, 0) as u64;
    if (HEADER_MIN as u64..=HEADER_MAX as u64).contains(&hl) && len >= 22 + 16 + 2 + hl {
        let mut m = [0u8; 8];
        f.seek(io::SeekFrom::Start(len - 2 - hl - 16))?;
        f.read_exact(&mut m)?;
        if &m == MARKER_V2 {
            let mut zl = [0u8; 8];
            f.read_exact(&mut zl)?;
            let zip_len = u64le(&zl, 0);
            if zip_len + 16 + 2 + hl > len {
                return Err(corrupt("declared zip length overruns the file"));
            }
            let mut header = vec![0u8; hl as usize];
            f.seek(io::SeekFrom::Start(len - 2 - hl))?;
            f.read_exact(&mut header)?;
            return Ok(Some(Tail::V2 { zip_len, header }));
        }
    }
    // v1: the last 16 bytes are `marker || u64 zip_len`. (A v1 file's final
    // two bytes are the top of that u64 — zero for any sane size — so the
    // v2 probe above never matches one.)
    let mut t = [0u8; 16];
    f.seek(io::SeekFrom::Start(len - 16))?;
    f.read_exact(&mut t)?;
    if &t[0..8] == MARKER_V1 {
        let zip_len = u64le(&t, 8);
        if zip_len + 16 > len {
            return Err(corrupt("declared zip length overruns the file"));
        }
        return Ok(Some(Tail::V1 { zip_len }));
    }
    Ok(None)
}

#[derive(Debug)]
pub struct Capsule {
    /// End of the immutable exe code region == start of the zip body.
    pub code_end: u64,
    /// Central-directory offset (relative to zip body start) and size.
    cd_offset: u64,
    cd_size: u64,
    pub entries: Vec<CapsuleEntry>,
    pub version: CapsuleVer,
    /// v2: the tail header bytes (a serialized key bundle; opaque here).
    pub header: Vec<u8>,
    /// v2: the decryption key, retained for reads. Zeroized on drop.
    key: Option<CapsuleKey>,
}

impl Drop for Capsule {
    fn drop(&mut self) {
        if let Some(k) = &mut self.key {
            k.zeroize();
        }
    }
}

/// Cheap presence/version check that needs no key — used at boot to decide
/// whether pre-UI hydration is possible (v1) or must wait for unlock (v2).
pub fn probe(exe: &Path) -> io::Result<Option<CapsuleVer>> {
    let mut f = fs::File::open(exe)?;
    let len = f.metadata()?.len();
    Ok(match read_tail(&mut f, len)? {
        Some(Tail::V1 { .. }) => Some(CapsuleVer::V1),
        Some(Tail::V2 { .. }) => Some(CapsuleVer::V2),
        None => None,
    })
}

/// Read a v2 exe's tail header (the wrapped-key bundle) without opening the
/// capsule. `Ok(None)` = no v2 capsule.
pub fn read_header(exe: &Path) -> io::Result<Option<Vec<u8>>> {
    let mut f = fs::File::open(exe)?;
    let len = f.metadata()?.len();
    Ok(match read_tail(&mut f, len)? {
        Some(Tail::V2 { header, .. }) => Some(header),
        _ => None,
    })
}

impl Capsule {
    /// Open the capsule at the tail of `exe`. `Ok(None)` = no capsule (a
    /// fresh build, nothing extracted). A v2 capsule REQUIRES `key` — without
    /// it even the entry names are unreadable (the error is PermissionDenied
    /// so callers can tell "locked" from "corrupt").
    pub fn open(exe: &Path, key: Option<&CapsuleKey>) -> io::Result<Option<Capsule>> {
        let mut f = fs::File::open(exe)?;
        let len = f.metadata()?.len();
        let (zip_len, version, header, key) = match read_tail(&mut f, len)? {
            None => return Ok(None),
            Some(Tail::V1 { zip_len }) => (zip_len, CapsuleVer::V1, Vec::new(), None),
            Some(Tail::V2 { zip_len, header }) => match key {
                Some(k) => (zip_len, CapsuleVer::V2, header, Some(*k)),
                None => return Err(io::Error::new(io::ErrorKind::PermissionDenied, SEALEDMSG)),
            },
        };
        let zip_start = len - 16 - zip_len - match version {
            CapsuleVer::V1 => 0,
            // marker(8) + zip_len(8) + header + u16
            CapsuleVer::V2 => 2 + header.len() as u64,
        };

        // EOCD is the last 22 bytes of the zip body.
        f.seek(io::SeekFrom::Start(zip_start + zip_len - 22))?;
        let mut eocd = [0u8; 22];
        f.read_exact(&mut eocd)?;
        if u32le(&eocd, 0) != SIG_EOCD {
            return Err(corrupt("EOCD signature"));
        }
        let n_entries = u16le(&eocd, 10) as usize;
        let cd_size = u32le(&eocd, 12) as u64;
        let cd_offset = u32le(&eocd, 16) as u64;
        if cd_offset + cd_size > zip_len {
            return Err(corrupt("central directory outside the zip body"));
        }

        f.seek(io::SeekFrom::Start(zip_start + cd_offset))?;
        let mut cd = vec![0u8; cd_size as usize];
        f.read_exact(&mut cd)?;

        let mut entries = Vec::with_capacity(n_entries);
        let mut p = 0usize;
        while p + 46 <= cd.len() {
            if u32le(&cd, p) != SIG_CD {
                return Err(corrupt("central directory signature"));
            }
            let method = u16le(&cd, p + 10);
            let crc = u32le(&cd, p + 16);
            let comp = u32le(&cd, p + 20) as u64;
            let uncomp = u32le(&cd, p + 24) as u64;
            let nlen = u16le(&cd, p + 28) as usize;
            let elen = u16le(&cd, p + 30) as usize;
            let clen = u16le(&cd, p + 32) as usize;
            let lho = u32le(&cd, p + 42) as u64;
            if p + 46 + nlen > cd.len() {
                return Err(corrupt("central directory name overruns"));
            }
            let name_raw = cd[p + 46..p + 46 + nlen].to_vec();
            let name = match (&key, version) {
                (Some(k), CapsuleVer::V2) => dec_name(k, &name_raw)?,
                _ => String::from_utf8_lossy(&name_raw).into_owned(),
            };
            entries.push(CapsuleEntry {
                name,
                name_raw,
                method,
                crc32: crc,
                comp_size: comp,
                uncomp_size: uncomp,
                local_offset: lho,
            });
            p += 46 + nlen + elen + clen;
        }
        if entries.len() != n_entries {
            return Err(corrupt("central directory entry count mismatch"));
        }
        Ok(Some(Capsule {
            code_end: zip_start,
            cd_offset,
            cd_size,
            entries,
            version,
            header,
            key,
        }))
    }

    pub fn entry(&self, name: &str) -> Option<&CapsuleEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn names(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.name.clone()).collect()
    }

    /// Read one entry fully into memory (small files only).
    pub fn read(&self, exe: &Path, name: &str) -> io::Result<Option<Vec<u8>>> {
        let Some(e) = self.entry(name) else {
            return Ok(None);
        };
        let mut f = fs::File::open(exe)?;
        seek_entry(&mut f, self.code_end, e)?;
        let mut out = Vec::new();
        stream_entry(&mut f, e, self.key.as_ref(), self.version, &mut out)?;
        Ok(Some(out))
    }

    /// Stream one entry to `dest` (parent dirs created). `Ok(false)` = the
    /// entry does not exist. Safe for large entries — nothing is buffered.
    pub fn extract(&self, exe: &Path, name: &str, dest: &Path) -> io::Result<bool> {
        let Some(e) = self.entry(name) else {
            return Ok(false);
        };
        let mut f = fs::File::open(exe)?;
        seek_entry(&mut f, self.code_end, e)?;
        if let Some(p) = dest.parent() {
            fs::create_dir_all(p)?;
        }
        let mut out = io::BufWriter::new(fs::File::create(dest)?);
        stream_entry(&mut f, e, self.key.as_ref(), self.version, &mut out)?;
        out.flush()?;
        out.get_ref().sync_all()?;
        Ok(true)
    }

    /// Stream every entry into `dest`, preserving virtual paths.
    pub fn extract_all(&self, exe: &Path, dest: &Path) -> io::Result<()> {
        for e in &self.entries {
            self.extract(exe, &e.name, &dest.join(&e.name))?;
        }
        Ok(())
    }

    /// Extract only entries MISSING from `dest` (returns how many). On v2
    /// this is the post-unlock hydration; when staging is already complete it
    /// costs one name-existence check per entry — no bytes move.
    pub fn extract_missing(&self, exe: &Path, dest: &Path) -> io::Result<usize> {
        let mut n = 0;
        for e in &self.entries {
            let out = dest.join(&e.name);
            if out.exists() {
                continue;
            }
            self.extract(exe, &e.name, &out)?;
            n += 1;
        }
        Ok(n)
    }
}

/// Collect the private state under `staging` as [`SealFile`]s (virtual path =
/// path relative to the staging root). Excluded: the WebView2 cache and any
/// `models/` tree (models live BESIDE the exe, never in the capsule — the
/// skip is defense for legacy configs that pointed the models dir into the
/// data root), leftover swap artifacts, and installer downloads.
pub fn collect_seal_files(staging: &Path) -> io::Result<Vec<SealFile>> {
    const SKIP_DIRS: &[&str] = &["webview", "models"];
    const SKIP_FILES: &[&str] = &["OllamaSetup.exe"];
    const TEXTY: &[&str] = &["toml", "log", "json", "md", "txt"];
    fn walk(dir: &Path, staging: &Path, out: &mut Vec<SealFile>) -> io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                if SKIP_DIRS.contains(&name.as_str()) {
                    continue;
                }
                walk(&path, staging, out)?;
            } else {
                if SKIP_FILES.contains(&name.as_str())
                    || name.ends_with(".exe.old")
                    || name.ends_with(".tmp")
                {
                    continue;
                }
                let rel = path
                    .strip_prefix(staging)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                let ext = path
                    .extension()
                    .map(|e| e.to_string_lossy().into_owned())
                    .unwrap_or_default()
                    .to_lowercase();
                out.push(SealFile {
                    name: rel,
                    source: path,
                    store: !TEXTY.contains(&ext.as_str()),
                });
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(staging, staging, &mut out)?;
    Ok(out)
}

/// How a seal writes the capsule tail.
pub enum SealMode<'a> {
    /// v1 plaintext tail (legacy exes that have not yet been unlocked by a
    /// Phase C build; also used by unit tests of the plain format).
    Plain,
    /// v2 sealed zone: every entry encrypted under `key`, `header` (a
    /// serialized key bundle holding the wrapped capsule key) written as the
    /// tail header.
    Sealed { key: &'a CapsuleKey, header: &'a [u8] },
}

/// The full seal cycle used at exit and by the auto-save: walk `staging`,
/// seal into `<exe>.new`, swap it in (safe while the exe runs).
pub fn seal_and_swap(exe: &Path, staging: &Path, mode: &SealMode<'_>) -> io::Result<SealOutcome> {
    let files = collect_seal_files(staging)?;
    let new = exe.with_extension("exe.new");
    let outcome = seal(exe, &files, &new, mode)?;
    swap_in_place(exe, &new)?;
    Ok(outcome)
}

/// Wait until the detached swap helper has moved `new_exe` into place (it
/// stops existing). Returns false on timeout — callers that plan to delete
/// the seal SOURCE must treat false as "do not delete".
pub fn wait_for_swap(new_exe: &Path, timeout_ms: u64) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
    while new_exe.exists() {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    true
}

/// Best-effort wipe of the staging dir after the final seal: the exe now
/// carries the private state, so the temp copy goes. `webview/` (Chromium
/// profile — UI prefs in localStorage, no app secrets) is kept, as is any
/// `models/` tree (models are NOT in the capsule — the staging copy is the
/// only copy, so it must never be deleted; normally models live beside the
/// exe and this is pure defense). Returns a note for the log.
pub fn wipe_staging(staging: &Path) -> String {
    const KEEP: &[&str] = &["webview", "models"];
    let mut left: Vec<String> = Vec::new();
    for attempt in 0..3 {
        left.clear();
        let Ok(rd) = fs::read_dir(staging) else {
            return "staging wipe: nothing to remove".into();
        };
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if KEEP.contains(&name.as_str()) {
                continue;
            }
            let p = entry.path();
            let ok = if p.is_dir() {
                fs::remove_dir_all(&p).is_ok()
            } else {
                fs::remove_file(&p).is_ok()
            };
            if !ok {
                left.push(name);
            }
        }
        if left.is_empty() {
            return "staging wiped (webview kept)".into();
        }
        if attempt < 2 {
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
    }
    format!("staging PARTIALLY wiped — locked files left: {}", left.join(", "))
}

/// Seek to an entry's payload (skips the local header + name + extra).
fn seek_entry(f: &mut fs::File, zip_start: u64, e: &CapsuleEntry) -> io::Result<()> {
    f.seek(io::SeekFrom::Start(zip_start + e.local_offset))?;
    let mut lh = [0u8; 30];
    f.read_exact(&mut lh)?;
    if u32le(&lh, 0) != SIG_LOCAL {
        return Err(corrupt("local header signature"));
    }
    let nlen = u16le(&lh, 26) as i64;
    let elen = u16le(&lh, 28) as i64;
    f.seek(io::SeekFrom::Current(nlen + elen))?;
    Ok(())
}

/// Stream one entry (decrypting + verifying for v2, CRC-checking for v1) to
/// `out`. Nothing larger than one chunk (or, for deflate, the whole
/// compressed payload — small state by contract) is held in memory.
fn stream_entry(
    f: &mut fs::File,
    e: &CapsuleEntry,
    key: Option<&CapsuleKey>,
    version: CapsuleVer,
    out: &mut impl Write,
) -> io::Result<()> {
    if version == CapsuleVer::V1 {
        stream_entry_plain(f, e, out)
    } else {
        let key = key.ok_or_else(|| corrupt(SEALEDMSG))?;
        stream_entry_sealed(f, e, key, out)
    }
}

fn stream_entry_plain(f: &mut fs::File, e: &CapsuleEntry, out: &mut impl Write) -> io::Result<()> {
    if e.method == 0 {
        let mut remaining = e.comp_size;
        let mut buf = vec![0u8; CHUNK.min(1 << 20)];
        let mut crc = crc32fast::Hasher::new();
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            let got = f.read(&mut buf[..want])?;
            if got == 0 {
                return Err(corrupt("unexpected EOF in stored entry"));
            }
            crc.update(&buf[..got]);
            out.write_all(&buf[..got])?;
            remaining -= got as u64;
        }
        if crc.finalize() != e.crc32 || e.comp_size != e.uncomp_size {
            return Err(corrupt("stored entry crc/size mismatch"));
        }
        Ok(())
    } else {
        // Deflate entries are small state by contract — decode in memory.
        let mut comp = vec![0u8; e.comp_size as usize];
        f.read_exact(&mut comp)?;
        let mut data = Vec::with_capacity(e.uncomp_size as usize);
        flate2::read::DeflateDecoder::new(&comp[..]).read_to_end(&mut data)?;
        verify_plain(e, &data)?;
        out.write_all(&data)
    }
}

fn stream_entry_sealed(
    f: &mut fs::File,
    e: &CapsuleEntry,
    key: &CapsuleKey,
    out: &mut impl Write,
) -> io::Result<()> {
    let mut prefix = [0u8; 8];
    f.read_exact(&mut prefix)?;
    let ct_total = e
        .comp_size
        .checked_sub(8)
        .ok_or_else(|| corrupt("sealed entry smaller than its nonce prefix"))?;
    // Chunk framing is derived from the ciphertext length alone as the loop
    // runs: every full chunk is CHUNK_PT + TAG bytes; only the last is
    // shorter.
    let cipher = gcm(key);
    let mut remaining = ct_total;
    let mut written: u64 = 0;
    let mut deflated: Option<Vec<u8>> = (e.method == 8).then(Vec::new);
    let mut idx: u32 = 0;
    while remaining > 0 {
        let want = remaining.min((CHUNK_PT + TAG) as u64) as usize;
        if want <= TAG {
            return Err(corrupt("sealed chunk has no plaintext"));
        }
        let mut buf = vec![0u8; want];
        f.read_exact(&mut buf)?;
        let pt = cipher
            .decrypt(Nonce::from_slice(&chunk_nonce(&prefix, idx)), &buf[..want])
            .map_err(|_| corrupt("entry failed authentication (wrong key or tampered exe)"))?;
        match &mut deflated {
            Some(acc) => acc.extend_from_slice(&pt),
            None => out.write_all(&pt)?,
        }
        written += pt.len() as u64;
        remaining -= want as u64;
        idx = idx.checked_add(1).ok_or_else(|| corrupt("chunk overflow"))?;
    }
    if let Some(acc) = &deflated {
        let mut data = Vec::with_capacity(e.uncomp_size as usize);
        flate2::read::DeflateDecoder::new(&acc[..]).read_to_end(&mut data)?;
        out.write_all(&data)?;
        written = data.len() as u64;
    }
    if written != e.uncomp_size {
        return Err(corrupt("entry size mismatch after decrypt"));
    }
    Ok(())
}

fn verify_plain(e: &CapsuleEntry, data: &[u8]) -> io::Result<()> {
    if data.len() as u64 != e.uncomp_size {
        return Err(corrupt("entry size mismatch"));
    }
    if crc32fast::hash(data) != e.crc32 {
        return Err(corrupt("entry crc mismatch"));
    }
    Ok(())
}

// ===== write side ==============================================================

/// One file to (re)write into the capsule this seal.
pub struct SealFile {
    /// Virtual path inside the capsule (e.g. "memory.db", "logs/app.log").
    pub name: String,
    /// Source file on disk (staging).
    pub source: PathBuf,
    /// true = STORED verbatim (ciphertext, dbs), false = DEFLATE (text).
    pub store: bool,
}

#[derive(Debug)]
pub struct SealOutcome {
    pub out_len: u64,
    pub entries: usize,
    /// Bytes appended beyond the previous artifact length.
    pub appended: u64,
    /// Whether this seal wrote a v2 (encrypted) capsule.
    pub sealed: bool,
}

/// Seal `files` into a v1 (plaintext) capsule at `out`. Legacy path.
pub fn seal_plain(exe: &Path, files: &[SealFile], out: &Path) -> io::Result<SealOutcome> {
    seal(exe, files, out, &SealMode::Plain)
}

/// Seal `files` into a fresh artifact at `out` = `[exe code region][new
/// capsule]`. The code region is carried over byte-identical from the current
/// `exe` (its length comes from the existing capsule, or the whole file on
/// the first seal); the old capsule is fully superseded — the capsule is
/// small private state, so every seal is a full rewrite (models are never
/// here; see the module docs). See [`SealMode`] for the two tail
/// generations. A v2 capsule is never re-sealed as v1 — refusing beats
/// silently downgrading the sealed zone to plain.
pub fn seal(exe: &Path, files: &[SealFile], out: &Path, mode: &SealMode<'_>) -> io::Result<SealOutcome> {
    let key = match mode {
        SealMode::Plain => None,
        SealMode::Sealed { key, header } => {
            if !(HEADER_MIN..=HEADER_MAX).contains(&header.len()) {
                return Err(corrupt(&format!(
                    "capsule header must be {HEADER_MIN}..={HEADER_MAX} bytes, got {}",
                    header.len()
                )));
            }
            Some((*key, *header))
        }
    };

    // The old capsule opens with the key when sealing v2 (the old entries
    // are not carried, but opening validates the marker/tail and yields the
    // code-region length).
    let old = Capsule::open(exe, key.map(|(k, _)| k))?;
    if key.is_none() && old.as_ref().is_some_and(|c| c.version == CapsuleVer::V2) {
        return Err(corrupt(
            "refusing to re-seal an encrypted capsule as plaintext",
        ));
    }
    // Defense in depth: a SEALED capsule is private state only — models
    // (public artifacts beside the exe) must never enter it. v1 keeps the
    // historical shape (it carried models) so legacy exes re-seal cleanly.
    if let (Some(_), Some(f)) = (
        key,
        files.iter().find(|f| f.name.starts_with("models/")),
    ) {
        return Err(corrupt(&format!(
            "refusing to seal {} — models live beside the exe, never in the capsule",
            f.name
        )));
    }
    let code_end = match &old {
        Some(c) => c.code_end,
        None => fs::metadata(exe)?.len(),
    };
    let prev_len = fs::metadata(exe)?.len();

    let mut src = fs::File::open(exe)?;
    let mut dst = io::BufWriter::new(fs::File::create(out)?);

    // 1. The immutable code region.
    copy_file_region(&mut src, &mut dst, 0, code_end)?;

    // 2. Fresh local records for the whole (small) private state.
    let mut zip_len: u64 = 0;
    let mut new_entries: Vec<CapsuleEntry> = Vec::new();
    for f in files {
        let entry = append_record(&mut dst, f, zip_len, key.map(|(k, _)| k))?;
        zip_len += entry.comp_size + 30 + entry.name_raw.len() as u64;
        new_entries.push(entry);
    }

    // 3. Central directory.
    let cd_offset = zip_len;
    let (time, date) = dos_now();
    let mut cd: Vec<u8> = Vec::new();
    let mut n: u16 = 0;
    for e in new_entries.iter() {
        let name = &e.name_raw;
        cd.extend_from_slice(&SIG_CD.to_le_bytes());
        cd.extend_from_slice(&20u16.to_le_bytes()); // version made by
        cd.extend_from_slice(&20u16.to_le_bytes()); // version needed
        cd.extend_from_slice(&0u16.to_le_bytes()); // flags
        cd.extend_from_slice(&e.method.to_le_bytes());
        cd.extend_from_slice(&time.to_le_bytes());
        cd.extend_from_slice(&date.to_le_bytes());
        cd.extend_from_slice(&e.crc32.to_le_bytes());
        cd.extend_from_slice(&(e.comp_size as u32).to_le_bytes());
        cd.extend_from_slice(&(e.uncomp_size as u32).to_le_bytes());
        cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes()); // extra len
        cd.extend_from_slice(&0u16.to_le_bytes()); // comment len
        cd.extend_from_slice(&0u16.to_le_bytes()); // disk number
        cd.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        cd.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        cd.extend_from_slice(&(e.local_offset as u32).to_le_bytes());
        cd.extend_from_slice(name);
        n += 1;
    }
    dst.write_all(&cd)?;

    // 4. EOCD + tail marker (v2 adds the wrapped-key header).
    let mut eocd = [0u8; 22];
    eocd[0..4].copy_from_slice(&SIG_EOCD.to_le_bytes());
    eocd[8..10].copy_from_slice(&n.to_le_bytes());
    eocd[10..12].copy_from_slice(&n.to_le_bytes());
    eocd[12..16].copy_from_slice(&(cd.len() as u32).to_le_bytes());
    eocd[16..20].copy_from_slice(&(cd_offset as u32).to_le_bytes());
    dst.write_all(&eocd)?;
    zip_len += cd.len() as u64 + 22;

    let tail_pad;
    if let Some((_, header)) = key {
        dst.write_all(MARKER_V2)?;
        dst.write_all(&zip_len.to_le_bytes())?;
        dst.write_all(header)?;
        let hl = header.len() as u16;
        dst.write_all(&hl.to_le_bytes())?;
        tail_pad = 16 + 2 + header.len() as u64;
    } else {
        dst.write_all(MARKER_V1)?;
        dst.write_all(&zip_len.to_le_bytes())?;
        tail_pad = 16;
    }

    dst.flush()?;
    let file = dst.into_inner()?;
    file.sync_all()?;

    let out_len = code_end + zip_len + tail_pad;
    Ok(SealOutcome {
        out_len,
        entries: n as usize,
        appended: out_len.saturating_sub(prev_len),
        sealed: key.is_some(),
    })
}

/// Write one local file record; returns the central-directory view of it.
fn append_record(
    dst: &mut impl Write,
    f: &SealFile,
    local_offset: u64,
    key: Option<&CapsuleKey>,
) -> io::Result<CapsuleEntry> {
    let (time, date) = dos_now();
    let name_raw: Vec<u8> = match key {
        Some(k) => enc_name(k, f.name.as_bytes())?,
        None => f.name.as_bytes().to_vec(),
    };
    let sealed = key.is_some();

    // Payload: for plain records the zip payload is the file bytes (deflated
    // for text); for sealed records the *plaintext* of the encryption is
    // that payload, then chunk-sealed — STORED entries stream chunk-by-chunk,
    // never in memory.
    if f.store {
        let pt_len = fs::metadata(&f.source)?.len();
        let (crc, comp, uncomp) = if sealed {
            (0u32, sealed_comp_size(pt_len), pt_len)
        } else {
            let (crc, len) = file_digest(&f.source)?;
            (crc, len, len)
        };
        let mut head = Vec::with_capacity(30 + name_raw.len());
        head.extend_from_slice(&SIG_LOCAL.to_le_bytes());
        head.extend_from_slice(&20u16.to_le_bytes());
        head.extend_from_slice(&0u16.to_le_bytes()); // flags
        head.extend_from_slice(&0u16.to_le_bytes()); // method: stored
        head.extend_from_slice(&time.to_le_bytes());
        head.extend_from_slice(&date.to_le_bytes());
        head.extend_from_slice(&crc.to_le_bytes());
        head.extend_from_slice(&(comp as u32).to_le_bytes());
        head.extend_from_slice(&(uncomp as u32).to_le_bytes());
        head.extend_from_slice(&(name_raw.len() as u16).to_le_bytes());
        head.extend_from_slice(&0u16.to_le_bytes()); // extra len
        head.extend_from_slice(&name_raw);
        dst.write_all(&head)?;

        let mut s = fs::File::open(&f.source)?;
        if let Some(k) = key {
            write_sealed_stream(&mut s, dst, k, pt_len)?;
        } else {
            copy_stream(&mut s, dst, comp)?;
        }
        Ok(CapsuleEntry {
            name: f.name.clone(),
            name_raw,
            method: 0,
            crc32: crc,
            comp_size: comp,
            uncomp_size: uncomp,
            local_offset,
        })
    } else {
        let data = fs::read(&f.source)?;
        let orig_len = data.len() as u64;
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(&data)?;
        let payload = enc.finish()?;
        let crc = if sealed { 0u32 } else { crc32fast::hash(&data) };
        let payload = if let Some(k) = key {
            enc_payload(k, &payload)?
        } else {
            payload
        };
        let mut head = Vec::with_capacity(30 + name_raw.len());
        head.extend_from_slice(&SIG_LOCAL.to_le_bytes());
        head.extend_from_slice(&20u16.to_le_bytes());
        head.extend_from_slice(&0u16.to_le_bytes()); // flags
        head.extend_from_slice(&8u16.to_le_bytes()); // method: deflate
        head.extend_from_slice(&time.to_le_bytes());
        head.extend_from_slice(&date.to_le_bytes());
        head.extend_from_slice(&crc.to_le_bytes());
        head.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        head.extend_from_slice(&(orig_len as u32).to_le_bytes());
        head.extend_from_slice(&(name_raw.len() as u16).to_le_bytes());
        head.extend_from_slice(&0u16.to_le_bytes()); // extra len
        head.extend_from_slice(&name_raw);
        dst.write_all(&head)?;
        dst.write_all(&payload)?;
        Ok(CapsuleEntry {
            name: f.name.clone(),
            name_raw,
            method: 8,
            crc32: crc,
            comp_size: payload.len() as u64,
            uncomp_size: orig_len,
            local_offset,
        })
    }
}

/// Stream exactly `pt_len` bytes of `src` into `dst` as sealed chunks
/// (prefix + GCM chunks). Single pass — the cipher runs as the bytes flow.
fn write_sealed_stream(
    src: &mut fs::File,
    dst: &mut impl Write,
    key: &CapsuleKey,
    pt_len: u64,
) -> io::Result<()> {
    let mut prefix = [0u8; 8];
    rand::thread_rng().fill_bytes(&mut prefix);
    dst.write_all(&prefix)?;
    let cipher = gcm(key);
    let mut buf = vec![0u8; CHUNK_PT];
    let mut remaining = pt_len;
    let mut idx: u32 = 0;
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        read_exact(src, &mut buf[..want])?;
        let ct = cipher
            .encrypt(Nonce::from_slice(&chunk_nonce(&prefix, idx)), &buf[..want])
            .map_err(|_| corrupt("payload encrypt"))?;
        dst.write_all(&ct)?;
        remaining -= want as u64;
        idx += 1;
    }
    Ok(())
}

/// `Read::read_exact` for a file — surfaces a distinguishable EOF error.
fn read_exact(src: &mut fs::File, buf: &mut [u8]) -> io::Result<()> {
    let mut filled = 0;
    while filled < buf.len() {
        let got = src.read(&mut buf[filled..])?;
        if got == 0 {
            return Err(corrupt("seal source shrank mid-copy"));
        }
        filled += got;
    }
    Ok(())
}

/// DOS date/time pair for "now" (zip records carry no timezone).
fn dos_now() -> (u16, u16) {
    use chrono::{Datelike, Timelike};
    let now = chrono::Local::now();
    let d = now.date_naive();
    let t = now.time();
    let date = (((d.year() - 1980) as u16) << 9) | ((d.month() as u16) << 5) | d.day() as u16;
    let time = ((t.hour() as u16) << 11) | ((t.minute() as u16) << 5) | (t.second() as u16 / 2);
    (time, date)
}

/// Streamed CRC + length of a file (pass one of a v1 STORED seal).
fn file_digest(path: &Path) -> io::Result<(u32, u64)> {
    let mut f = fs::File::open(path)?;
    let mut crc = crc32fast::Hasher::new();
    let mut len = 0u64;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let got = f.read(&mut buf)?;
        if got == 0 {
            break;
        }
        crc.update(&buf[..got]);
        len += got as u64;
    }
    Ok((crc.finalize(), len))
}

/// Copy `[from, from+len)` of an open file (seeks first).
fn copy_file_region(
    src: &mut fs::File,
    dst: &mut impl Write,
    from: u64,
    len: u64,
) -> io::Result<u64> {
    src.seek(io::SeekFrom::Start(from))?;
    copy_stream(src, dst, len)
}

/// Copy exactly `len` bytes from a reader (chunked).
fn copy_stream(src: &mut impl Read, dst: &mut impl Write, len: u64) -> io::Result<u64> {
    let mut remaining = len;
    let mut buf = vec![0u8; CHUNK];
    let mut copied = 0u64;
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        let got = src.read(&mut buf[..want])?;
        if got == 0 {
            return Err(corrupt("unexpected EOF during copy"));
        }
        dst.write_all(&buf[..got])?;
        remaining -= got as u64;
        copied += got as u64;
    }
    Ok(copied)
}

// ===== compaction =============================================================

/// Full rewrite keeping only live private entries — drops dead bytes and
/// any `models/` entries a legacy v1 capsule may still carry (models moved
/// out of the capsule; they are NOT extracted, their exe-dir folder is the
/// authority). Streams through `scratch`.
pub fn compact(exe: &Path, out: &Path, scratch: &Path, mode: &SealMode<'_>) -> io::Result<u64> {
    let key = match mode {
        SealMode::Plain => None,
        SealMode::Sealed { key, .. } => Some(*key),
    };
    let Some(c) = Capsule::open(exe, key)? else {
        return Ok(fs::metadata(exe).map(|m| m.len()).unwrap_or(0));
    };
    fs::create_dir_all(scratch)?;
    let mut files = Vec::new();
    for (i, e) in c.entries.iter().enumerate() {
        if e.name.starts_with("models/") {
            continue; // legacy capsule payload — models live beside the exe now
        }
        let staging = scratch.join(format!("{i:04}"));
        c.extract(exe, &e.name, &staging)?;
        files.push(SealFile {
            name: e.name.clone(),
            source: staging,
            store: e.method == 0,
        });
    }
    let outcome = seal(exe, &files, out, mode)?;
    let _ = fs::remove_dir_all(scratch);
    Ok(outcome.out_len)
}

// ===== live swap ==============================================================

/// Replace `current` (possibly a RUNNING exe) with `new` via a detached cmd
/// helper: rename current→`.exe.old` (renaming a running exe is allowed),
/// move new→current, best-effort delete of the old copy (locked while the
/// renaming process lives — swept by [`sweep_old`] on a later launch).
pub fn swap_in_place(current: &Path, new: &Path) -> io::Result<()> {
    let old = current.with_extension("exe.old");
    let cur = current.to_string_lossy().to_string();
    let old_name = old
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "app.exe.old".into());
    let ol = old.to_string_lossy().to_string();
    let nw = new.to_string_lossy().to_string();
    let log = std::env::temp_dir().join("phxcaps-swap.log");
    let log_s = log.to_string_lossy().to_string();
    let script = format!(
        "(ren \"{cur}\" \"{old_name}\" & move /Y \"{nw}\" \"{cur}\" & (ping -n 3 127.0.0.1 >nul & del \"{ol}\")) >>\"{log_s}\" 2>&1"
    );
    let mut cmd = std::process::Command::new("cmd");
    // cmd.exe needs the script VERBATIM: normal arg passing backslash-escapes
    // the inner quotes, which cmd does not understand (the script dies
    // silently). raw_arg is the correct vehicle on Windows.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW); // hidden console, survives our exit
        cmd.raw_arg("/C");
        cmd.raw_arg(&script);
    }
    #[cfg(not(windows))]
    {
        cmd.args(["/C", &script]);
    }
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    cmd.spawn()?;
    Ok(())
}

/// Delete a leftover `.exe.old` from a previous swap.
pub fn sweep_old(exe: &Path) {
    let _ = fs::remove_file(exe.with_extension("exe.old"));
}

// ===== tests ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("phxcaps-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn fake_exe(d: &Path, size: usize) -> PathBuf {
        let p = d.join("app.exe");
        let mut bytes = vec![0u8; size];
        bytes[0] = b'M';
        bytes[1] = b'Z';
        for (i, b) in bytes.iter_mut().enumerate().skip(2) {
            *b = (i % 251) as u8;
        }
        fs::write(&p, bytes).unwrap();
        p
    }

    fn writef(d: &Path, name: &str, contents: &[u8]) -> PathBuf {
        let p = d.join(name);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&p, contents).unwrap();
        p
    }

    fn key() -> CapsuleKey {
        let mut k = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut k);
        k
    }

    fn header() -> Vec<u8> {
        b"phoenix-keys-v1\nprimary_salt=AAAAAAAAAAAAAAAAAAAAAA==\nprimary_blob=AAAA\n".to_vec()
    }

    // -- v1 (plain) ------------------------------------------------------------

    #[test]
    fn fresh_exe_has_no_capsule() {
        let d = dir("fresh");
        let exe = fake_exe(&d, 1024);
        assert!(Capsule::open(&exe, None).unwrap().is_none());
        assert_eq!(probe(&exe).unwrap(), None);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn first_seal_roundtrip() {
        let d = dir("first");
        let exe = fake_exe(&d, 4096);
        let txt = b"hello capsule ".repeat(200);
        let bin: Vec<u8> = (0u32..16_384).map(|i| i as u8).collect::<Vec<_>>();
        let out = d.join("app.new");
        seal_plain(
            &exe,
            &[
                SealFile { name: "logs/a.txt".into(), source: writef(&d, "a.txt", &txt), store: false },
                SealFile { name: "data/b.bin".into(), source: writef(&d, "b.bin", &bin), store: true },
            ],
            &out,
        )
        .unwrap();
        fs::copy(&out, &exe).unwrap(); // the swap "happened"

        let cap = Capsule::open(&exe, None).unwrap().expect("capsule present");
        assert_eq!(cap.version, CapsuleVer::V1);
        assert_eq!(cap.code_end, 4096, "code region is the original exe");
        assert_eq!(cap.names().len(), 2);
        assert_eq!(cap.read(&exe, "logs/a.txt").unwrap().unwrap(), txt);
        assert_eq!(cap.read(&exe, "data/b.bin").unwrap().unwrap(), bin);

        // Extraction to paths preserves virtual directories.
        let dest = d.join("x");
        cap.extract_all(&exe, &dest).unwrap();
        assert_eq!(fs::read(dest.join("data/b.bin")).unwrap(), bin);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn collect_excludes_models_and_webview() {
        let d = dir("collect");
        let stage = d.join("stage");
        writef(&stage, "memory.db", b"db");
        writef(&stage, "models/qwen/x.gguf", b"gguf");
        writef(&stage, "models/qwen/x.tokenizer.json", b"tk");
        writef(&stage, "webview/cache.dat", b"cache");
        writef(&stage, "logs/app.log", b"log");
        let names: Vec<String> = collect_seal_files(&stage)
            .unwrap()
            .into_iter()
            .map(|f| f.name)
            .collect();
        assert!(names.contains(&"memory.db".to_string()));
        assert!(names.contains(&"logs/app.log".to_string()));
        assert!(
            !names.iter().any(|n| n.starts_with("models/")),
            "models must never be collected: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("webview/")),
            "webview must never be collected: {names:?}"
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn seal_refuses_model_paths() {
        let d = dir("nomodels");
        let exe = fake_exe(&d, 1024);
        let err = seal(
            &exe,
            &[SealFile { name: "models/x.gguf".into(), source: writef(&d, "x", b"x"), store: true }],
            &d.join("app.new"),
            &SealMode::Sealed { key: &key(), header: &header() },
        )
        .unwrap_err();
        assert!(err.to_string().contains("never in the capsule"), "{err}");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn legacy_model_entries_are_dropped_on_reseal() {
        // A pre-Phase-C exe may still carry models in its v1 capsule. Any
        // later seal (plain or sealed) drops them — models belong beside the
        // exe, and the old records ride away as unlisted bytes.
        let d = dir("legacy");
        let exe = fake_exe(&d, 1024);
        let model = b"FAKE-GGUF-BYTES".repeat(1000);
        seal_plain(
            &exe,
            &[
                SealFile { name: "models/old.gguf".into(), source: writef(&d, "old.gguf", &model), store: true },
                SealFile { name: "memory.db".into(), source: writef(&d, "db", b"v1"), store: true },
            ],
            &d.join("app.new"),
        )
        .unwrap();
        fs::copy(d.join("app.new"), &exe).unwrap();
        assert!(Capsule::open(&exe, None).unwrap().unwrap().names().contains(&"models/old.gguf".to_string()));

        // v1 re-seal (exit without unlock): models dropped.
        seal_plain(
            &exe,
            &[SealFile { name: "memory.db".into(), source: writef(&d, "db", b"v2"), store: true }],
            &d.join("app.new"),
        )
        .unwrap();
        fs::copy(d.join("app.new"), &exe).unwrap();
        let cap = Capsule::open(&exe, None).unwrap().unwrap();
        assert_eq!(cap.names(), vec!["memory.db".to_string()]);
        assert_eq!(cap.read(&exe, "memory.db").unwrap().unwrap(), b"v2");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn swap_replaces_files() {
        let d = dir("swap");
        let cur = writef(&d, "app.exe", b"OLD-CONTENT");
        let new = writef(&d, "app.exe.new", b"NEW-CONTENT");
        swap_in_place(&cur, &new).unwrap();
        for _ in 0..50 {
            std::thread::sleep(std::time::Duration::from_millis(100));
            if fs::read(&cur).map(|b| b == b"NEW-CONTENT").unwrap_or(false) {
                let _ = fs::remove_dir_all(&d);
                return;
            }
        }
        let _ = fs::remove_dir_all(&d);
        panic!("swap helper did not complete in time");
    }

    // -- v2 (sealed) -----------------------------------------------------------

    #[test]
    fn v2_roundtrip_and_opacity() {
        let d = dir("v2rt");
        let exe = fake_exe(&d, 4096);
        let k = key();
        let hdr = header();
        let txt = b"secret state ".repeat(300);
        let bin: Vec<u8> = (0u32..40_000).map(|i| (i * 31 % 256) as u8).collect::<Vec<_>>();

        seal(
            &exe,
            &[
                SealFile { name: "config.toml".into(), source: writef(&d, "cfg", &txt), store: false },
                SealFile { name: "vault/blob.bin".into(), source: writef(&d, "blob", &bin), store: true },
            ],
            &d.join("app.new"),
            &SealMode::Sealed { key: &k, header: &hdr },
        )
        .unwrap();
        fs::copy(d.join("app.new"), &exe).unwrap();

        // Probe + header without the key.
        assert_eq!(probe(&exe).unwrap(), Some(CapsuleVer::V2));
        assert_eq!(read_header(&exe).unwrap().unwrap(), hdr);

        // Opening without the key is refused (locked, not corrupt).
        let err = Capsule::open(&exe, None).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);

        // With the key: names decrypt, payloads round-trip.
        let cap = Capsule::open(&exe, Some(&k)).unwrap().expect("capsule");
        assert_eq!(cap.version, CapsuleVer::V2);
        assert_eq!(
            cap.names(),
            vec!["config.toml".to_string(), "vault/blob.bin".to_string()]
        );
        assert_eq!(cap.read(&exe, "config.toml").unwrap().unwrap(), txt);
        assert_eq!(cap.read(&exe, "vault/blob.bin").unwrap().unwrap(), bin);

        // Opacity: neither the plaintext names nor the payloads appear in
        // the raw file (beyond the code region).
        let raw = fs::read(&exe).unwrap();
        let tail = &raw[4096..];
        let find = |hay: &[u8], needle: &[u8]| {
            hay.windows(needle.len()).any(|w| w == needle)
        };
        assert!(!find(tail, b"config.toml"), "entry name must not be plaintext");
        assert!(!find(tail, b"vault/"), "entry name must not be plaintext");
        assert!(!find(tail, &txt[..64]), "text payload must not be plaintext");
        assert!(
            !find(tail, &bin[..64]) && !find(tail, &bin[20_000..20_064]),
            "binary payload must not be plaintext"
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn v2_wrong_key_fails_cleanly() {
        let d = dir("v2wrong");
        let exe = fake_exe(&d, 2048);
        let k = key();
        seal(
            &exe,
            &[SealFile { name: "s.txt".into(), source: writef(&d, "s.txt", b"state"), store: false }],
            &d.join("app.new"),
            &SealMode::Sealed { key: &k, header: &header() },
        )
        .unwrap();
        fs::copy(d.join("app.new"), &exe).unwrap();

        let bad = key();
        assert_ne!(bad, k);
        // Wrong key: the very first entry NAME fails authentication.
        assert!(Capsule::open(&exe, Some(&bad)).is_err());
        // And with the right key but a corrupted payload byte, the read fails
        // the GCM tag — flip one ciphertext byte inside the first entry.
        let cap = Capsule::open(&exe, Some(&k)).unwrap().unwrap();
        let e = cap.entry("s.txt").unwrap().clone();
        let at = (cap.code_end + e.local_offset + 30 + e.name_raw.len() as u64 + 9) as usize;
        let mut raw = fs::read(&exe).unwrap();
        raw[at] ^= 0xFF;
        fs::write(&exe, raw).unwrap();
        let cap2 = Capsule::open(&exe, Some(&k)).unwrap().unwrap();
        let err = cap2.read(&exe, "s.txt").unwrap_err();
        assert!(err.to_string().contains("authentication"), "{err}");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn v2_chunk_boundaries() {
        // Entries at/around the 1 MiB chunk seams: the framing math (comp
        // from ciphertext length alone) must hold at every boundary.
        let d = dir("v2chunks");
        let exe = fake_exe(&d, 1024);
        let k = key();
        for (i, size) in [
            0usize,
            1,
            CHUNK_PT - 1,
            CHUNK_PT,
            CHUNK_PT + 1,
            2 * CHUNK_PT,
            2 * CHUNK_PT + 7,
        ]
        .iter()
        .enumerate()
        {
            let data: Vec<u8> = (0..*size).map(|j| (j % 251) as u8).collect();
            let name = format!("vault/{i}.dat");
            seal(
                &exe,
                &[SealFile { name: name.clone(), source: writef(&d, &format!("{i}.dat"), &data), store: true }],
                &d.join("app.new"),
                &SealMode::Sealed { key: &k, header: &header() },
            )
            .unwrap();
            fs::copy(d.join("app.new"), &exe).unwrap();
            let cap = Capsule::open(&exe, Some(&k)).unwrap().unwrap();
            assert_eq!(
                cap.read(&exe, &name).unwrap().unwrap(),
                data,
                "size {size} must round-trip"
            );
        }
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn v1_to_v2_migration_seals_private_state_only() {
        let d = dir("v2mig");
        let exe = fake_exe(&d, 2048);
        let model = b"FAKE-GGUF".repeat(10_000);
        let state = b"plain state v1".repeat(100);

        // A v1 exe with state AND a legacy model inside its capsule.
        seal_plain(
            &exe,
            &[
                SealFile { name: "models/m.gguf".into(), source: writef(&d, "m.bin", &model), store: true },
                SealFile { name: "state.toml".into(), source: writef(&d, "st", state.as_slice()), store: false },
            ],
            &d.join("app.new"),
        )
        .unwrap();
        fs::copy(d.join("app.new"), &exe).unwrap();

        // The migration seal (v1 present, Sealed mode).
        let k = key();
        seal(
            &exe,
            &[SealFile { name: "state.toml".into(), source: writef(&d, "st", b"plain state v2".repeat(100).as_slice()), store: false }],
            &d.join("app.new"),
            &SealMode::Sealed { key: &k, header: &header() },
        )
        .unwrap();
        fs::copy(d.join("app.new"), &exe).unwrap();

        assert_eq!(probe(&exe).unwrap(), Some(CapsuleVer::V2));
        let cap = Capsule::open(&exe, Some(&k)).unwrap().unwrap();
        assert_eq!(cap.names(), vec!["state.toml".to_string()], "model dropped");
        assert!(cap.read(&exe, "state.toml").unwrap().unwrap().starts_with(b"plain state v2"));

        // The private state is gone from the clear; the model bytes were
        // dropped entirely (not carried, not re-encrypted).
        let after = fs::read(&exe).unwrap();
        let find = |hay: &[u8], needle: &[u8]| hay.windows(needle.len()).any(|w| w == needle);
        assert!(!find(&after[2048..], b"state.toml"));
        assert!(!find(&after[2048..], &b"plain state v2".repeat(100)[..64]));
        assert!(!find(&after[2048..], &model[..64]), "model bytes must be gone");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn v2_never_downgrades_to_plain() {
        let d = dir("v2nodown");
        let exe = fake_exe(&d, 1024);
        let k = key();
        seal(
            &exe,
            &[SealFile { name: "s.txt".into(), source: writef(&d, "s.txt", b"x"), store: false }],
            &d.join("app.new"),
            &SealMode::Sealed { key: &k, header: &header() },
        )
        .unwrap();
        fs::copy(d.join("app.new"), &exe).unwrap();
        assert!(seal_plain(
            &exe,
            &[SealFile { name: "s.txt".into(), source: writef(&d, "s.txt", b"y"), store: false }],
            &d.join("app.new"),
        )
        .is_err());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn v2_compact_drops_dead_bytes() {
        let d = dir("v2cmp");
        let exe = fake_exe(&d, 1024);
        let k = key();
        for i in 0..3 {
            let body = format!("sealed state {i}").repeat(50);
            seal(
                &exe,
                &[SealFile { name: "state.txt".into(), source: writef(&d, "s.txt", body.as_bytes()), store: false }],
                &d.join("app.new"),
                &SealMode::Sealed { key: &k, header: &header() },
            )
            .unwrap();
            fs::copy(d.join("app.new"), &exe).unwrap();
        }
        let before = fs::metadata(&exe).unwrap().len();
        compact(&exe, &d.join("app.cmp"), &d.join("scratch"), &SealMode::Sealed { key: &k, header: &header() }).unwrap();
        fs::copy(d.join("app.cmp"), &exe).unwrap();
        let after = fs::metadata(&exe).unwrap().len();
        // Full-rewrite seals leave no dead bytes, so compaction is now a
        // maintenance pass: same-or-smaller, content intact, still sealed.
        assert!(after <= before, "compaction must not grow: {before} -> {after}");
        let cap = Capsule::open(&exe, Some(&k)).unwrap().unwrap();
        assert_eq!(cap.version, CapsuleVer::V2);
        assert!(cap.read(&exe, "state.txt").unwrap().unwrap().starts_with(b"sealed state 2"));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn v2_extract_all_and_missing() {
        let d = dir("v2ex");
        let exe = fake_exe(&d, 1024);
        let k = key();
        let data = b"some log line\n".repeat(100);
        seal(
            &exe,
            &[
                SealFile { name: "logs/a.log".into(), source: writef(&d, "a.log", &data), store: false },
                SealFile { name: "b.bin".into(), source: writef(&d, "b.bin", &data[..500]), store: true },
            ],
            &d.join("app.new"),
            &SealMode::Sealed { key: &k, header: &header() },
        )
        .unwrap();
        fs::copy(d.join("app.new"), &exe).unwrap();

        let dest = d.join("stage");
        let cap = Capsule::open(&exe, Some(&k)).unwrap().unwrap();
        let n = cap.extract_missing(&exe, &dest).unwrap();
        assert_eq!(n, 2);
        assert_eq!(fs::read(dest.join("logs/a.log")).unwrap(), data);
        // Second pass: staging complete — nothing moves.
        let n2 = cap.extract_missing(&exe, &dest).unwrap();
        assert_eq!(n2, 0);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn wipe_keeps_webview_and_models() {
        let d = dir("wipe");
        let stage = d.join("stage");
        writef(&stage, "memory.db", b"db");
        writef(&stage, "logs/app.log", b"log");
        writef(&stage, "webview/cache.dat", b"cache");
        writef(&stage, "models/x/y.gguf", b"gguf");
        let note = wipe_staging(&stage);
        assert!(note.contains("wiped"), "{note}");
        assert!(!stage.join("memory.db").exists());
        assert!(!stage.join("logs").exists());
        assert!(stage.join("webview/cache.dat").exists(), "webview kept");
        assert!(stage.join("models/x/y.gguf").exists(), "models kept");
        let _ = fs::remove_dir_all(&d);
    }
}
