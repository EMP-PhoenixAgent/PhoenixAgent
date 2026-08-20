//! Passphrase → 32-byte key via Argon2id, with persisted random salt.
//!
//! The salt is stored unencrypted (it need not be secret), while the derived
//! key lives only in process memory and is zeroized on drop. The database file
//! on disk remains opaque ciphertext without this key.
//!
//! Optional 2FA via RFC 6238 TOTP lives in the [`totp`] submodule.

use std::path::Path;

pub mod totp;

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use argon2::{Algorithm, Argon2, Params, Version};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use rand::RngCore;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{PhoenixError, Result};

/// Argon2 salt length in bytes.
pub const SALT_LEN: usize = 16;
/// Derived key length: 256 bits for AES-256 (SQLCipher default).
pub const KEY_LEN: usize = 32;

/// A 32-byte derived key. Zeroized from memory when dropped.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct DerivedKey(pub [u8; KEY_LEN]);

impl DerivedKey {
    /// Hex-encode the key for SQLCipher's `PRAGMA key = "x'<hex>'"`.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

/// Load the salt from disk, generating and persisting a fresh one if absent.
///
/// This is called on first run (init) and on every subsequent unlock.
pub fn load_or_create_salt(path: &Path) -> Result<[u8; SALT_LEN]> {
    if path.exists() {
        let bytes = std::fs::read(path)?;
        if bytes.len() != SALT_LEN {
            return Err(PhoenixError::Crypto(format!(
                "salt file {} is {} bytes, expected {SALT_LEN}; delete it to reinitialize",
                path.display(),
                bytes.len()
            )));
        }
        let mut salt = [0u8; SALT_LEN];
        salt.copy_from_slice(&bytes);
        Ok(salt)
    } else {
        let mut salt = [0u8; SALT_LEN];
        rand::thread_rng().fill_bytes(&mut salt);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, salt)?;
        Ok(salt)
    }
}

/// Derive a 32-byte key from a passphrase + salt using Argon2id.
///
/// Parameters: 64 MiB memory, 3 passes, 4 lanes — strong but ~0.3s on modern HW.
///
/// If `totp_code` is provided (2FA enabled), it is appended to the passphrase
/// with a non-printable separator before derivation, so both the passphrase and
/// a valid time-based code are required to unlock.
///
/// **Security note:** because the TOTP seed is stored inside the encrypted DB,
/// 2FA here raises the cost of *offline passphrase guessing* (an attacker must
/// also brute-force the 6-digit window per guess) rather than being a true
/// server-side "something you have" factor. It is still a meaningful upgrade
/// over passphrase-alone. See `totp` module docs.
pub fn derive_key(
    passphrase: &str,
    salt: &[u8; SALT_LEN],
    totp_code: Option<&str>,
) -> Result<DerivedKey> {
    // Fold the optional 2FA code into the derived material. The separator is a
    // non-printable byte so the code can't be confused with a passphrase suffix.
    let effective: Vec<u8> = match totp_code {
        Some(code) => {
            let mut v = passphrase.as_bytes().to_vec();
            v.push(0x1F); // ASCII Unit Separator
            v.extend_from_slice(code.trim().as_bytes());
            v
        }
        None => passphrase.as_bytes().to_vec(),
    };

    let params = Params::new(64 * 1024, 3, 4, Some(KEY_LEN))
        .map_err(|e| PhoenixError::Crypto(format!("argon2 params: {e}")))?;
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

    let mut out = [0u8; KEY_LEN];
    argon2
        .hash_password_into(&effective, salt, &mut out)
        .map_err(|e| PhoenixError::Crypto(format!("argon2 derive: {e}")))?;

    Ok(DerivedKey(out))
}

/// Generate and persist a fresh random salt, returning it. Used during
/// passphrase changes to rotate the salt alongside the key.
pub fn rotate_salt(path: &Path) -> Result<[u8; SALT_LEN]> {
    let mut salt = [0u8; SALT_LEN];
    rand::thread_rng().fill_bytes(&mut salt);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, salt)?;
    Ok(salt)
}

// ===========================================================================
// Key wrapping — the two-password model
// ===========================================================================
//
// The SQLCipher DB key (`DerivedKey`) is a long random-ish value derived once
// from the *DB password* (default "PhoenixAgent", set at first run, then the
// agent uses it autonomously). So the user doesn't have to remember it, the
// `DerivedKey` is stored on disk **encrypted (wrapped)** under a key derived
// from the user's *launch password*. Recovery, when 2FA is enabled, is a second
// wrap of the same `DerivedKey` under a key derived from the 2FA TOTP code.
//
// Wrapping uses AES-256-GCM (authenticated): a wrong launch password or a
// tampered blob fails the GCM tag check and errors cleanly — it never yields a
// bogus key that would silently corrupt the DB.

/// AES-GCM nonce length (96 bits, the GCM standard).
const NONCE_LEN: usize = 12;

/// Derive a 32-byte *wrap key* from a passphrase (the launch password or a
/// TOTP recovery code) + salt via Argon2id. Same cost as `derive_key` but kept
/// separate so wrap keys and the SQLCipher key can never be confused.
pub fn derive_wrap_key(passphrase: &str, salt: &[u8; SALT_LEN]) -> Result<[u8; KEY_LEN]> {
    let params = Params::new(64 * 1024, 3, 4, Some(KEY_LEN))
        .map_err(|e| PhoenixError::Crypto(format!("argon2 params: {e}")))?;
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut out = [0u8; KEY_LEN];
    argon2
        .hash_password_into(passphrase.as_bytes(), salt, &mut out)
        .map_err(|e| PhoenixError::Crypto(format!("argon2 wrap-key derive: {e}")))?;
    Ok(out)
}

/// Seal a 32-byte key under a 32-byte wrap key. Returns `nonce(12) || ct+tag(48)`.
pub fn wrap_key(plaintext: &DerivedKey, wrap_key: &[u8; KEY_LEN]) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(wrap_key));
    let mut nonce_bytes = [0u8; NONCE_LEN];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ct = cipher
        .encrypt(nonce, plaintext.0.as_ref())
        .map_err(|e| PhoenixError::Crypto(format!("aes-gcm seal: {e}")))?;
    let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ct);
    Ok(out)
}

/// Open a wrap blob under a 32-byte wrap key, returning the plaintext key.
/// Fails on a wrong key or tampering (GCM tag mismatch) — never yields garbage.
pub fn unwrap_key(blob: &[u8], wrap_key: &[u8; KEY_LEN]) -> Result<DerivedKey> {
    if blob.len() < NONCE_LEN {
        return Err(PhoenixError::Crypto("key wrap blob too short".into()));
    }
    let (nonce_bytes, ct) = blob.split_at(NONCE_LEN);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(wrap_key));
    let pt = cipher
        .decrypt(Nonce::from_slice(nonce_bytes), ct)
        .map_err(|_| PhoenixError::Crypto("invalid launch password or tampered key wrap".into()))?;
    if pt.len() != KEY_LEN {
        return Err(PhoenixError::Crypto(format!(
            "unwrapped key is {} bytes, expected {KEY_LEN}",
            pt.len()
        )));
    }
    let mut arr = [0u8; KEY_LEN];
    arr.copy_from_slice(&pt);
    Ok(DerivedKey(arr))
}

/// Generate a fresh random 16-byte salt (for wrap-key derivation). Does NOT
/// persist — callers persist it inside the key bundle.
pub fn random_salt() -> [u8; SALT_LEN] {
    let mut salt = [0u8; SALT_LEN];
    rand::thread_rng().fill_bytes(&mut salt);
    salt
}

// ---------------------------------------------------------------------------
// On-disk key bundle (`keys.phx`) — holds the wrapped DB key(s).
// ---------------------------------------------------------------------------

/// Magic header line identifying a v1 key bundle.
const KEY_BUNDLE_MAGIC: &str = "phoenix-keys-v1";

/// The on-disk bundle of wrapped keys. The DB `DerivedKey` is never stored in
/// plaintext; it lives only inside these wraps.
#[derive(Debug, Clone)]
pub struct KeyBundle {
    /// Salt used to derive the primary wrap key (from the launch password).
    pub primary_salt: [u8; SALT_LEN],
    /// DB key wrapped under the launch password's wrap key.
    pub primary_blob: Vec<u8>,
    /// When 2FA is enabled: the TOTP seed (base32) used to (a) verify a typed
    /// recovery code and (b) derive the recovery wrap key. Stored unencrypted
    /// — it is a recovery secret, not the data key, and must be readable pre-DB.
    pub recovery_seed_b32: Option<String>,
    /// Salt for the recovery wrap key (derived from the TOTP seed).
    pub recovery_salt: Option<[u8; SALT_LEN]>,
    /// DB key wrapped under the recovery wrap key (present only with 2FA).
    pub recovery_blob: Option<Vec<u8>>,
}

impl KeyBundle {
    /// Build a fresh bundle wrapping `db_key` under the launch password.
    /// Generates a new primary salt. No recovery wrap (call `set_recovery`).
    pub fn create(db_key: &DerivedKey, launch_password: &str) -> Result<Self> {
        let primary_salt = random_salt();
        let wk = derive_wrap_key(launch_password, &primary_salt)?;
        let primary_blob = wrap_key(db_key, &wk)?;
        Ok(Self {
            primary_salt,
            primary_blob,
            recovery_seed_b32: None,
            recovery_salt: None,
            recovery_blob: None,
        })
    }

    /// Unwrap the DB key using the launch password. Errors on a wrong password.
    pub fn unwrap_primary(&self, launch_password: &str) -> Result<DerivedKey> {
        let wk = derive_wrap_key(launch_password, &self.primary_salt)?;
        unwrap_key(&self.primary_blob, &wk)
    }

    /// Add/replace the recovery wrap. The DB key is encrypted under a wrap key
    /// derived from the TOTP **seed** (stable), and the seed is stored so a
    /// typed recovery **code** can be verified before unwrapping. Requires the
    /// plaintext DB key.
    pub fn set_recovery(&mut self, db_key: &DerivedKey, totp_seed_b32: &str) -> Result<()> {
        let salt = random_salt();
        let wk = derive_wrap_key(totp_seed_b32, &salt)?;
        let blob = wrap_key(db_key, &wk)?;
        self.recovery_seed_b32 = Some(totp_seed_b32.to_string());
        self.recovery_salt = Some(salt);
        self.recovery_blob = Some(blob);
        Ok(())
    }

    /// Remove the recovery wrap (called when 2FA is disabled).
    pub fn clear_recovery(&mut self) {
        self.recovery_seed_b32 = None;
        self.recovery_salt = None;
        self.recovery_blob = None;
    }

    /// Whether a recovery wrap is present (2FA was enabled).
    pub fn has_recovery(&self) -> bool {
        self.recovery_blob.is_some()
    }

    /// Verify a typed TOTP code against the stored recovery seed, returning the
    /// underlying TOTP instance if it matches. The caller uses this to gate the
    /// unwrap. Errors if no recovery seed or the code is invalid.
    pub fn verify_recovery_code(&self, code: &str) -> Result<()> {
        let seed = self
            .recovery_seed_b32
            .as_ref()
            .ok_or_else(|| PhoenixError::Crypto("no recovery wrap (2FA not enabled)".into()))?;
        // Delegate to the totp submodule for verification.
        crate::crypto::totp::verify_raw(seed, code)
    }

    /// Unwrap the DB key via the recovery wrap. The caller MUST have already
    /// verified a current TOTP code via [`verify_recovery_code`]; the wrap key
    /// is derived from the (constant) seed, not the code.
    pub fn unwrap_recovery(&self) -> Result<DerivedKey> {
        let salt = self
            .recovery_salt
            .ok_or_else(|| PhoenixError::Crypto("no recovery wrap (2FA not enabled)".into()))?;
        let blob = self
            .recovery_blob
            .as_ref()
            .ok_or_else(|| PhoenixError::Crypto("no recovery wrap (2FA not enabled)".into()))?;
        let seed = self
            .recovery_seed_b32
            .as_ref()
            .ok_or_else(|| PhoenixError::Crypto("no recovery wrap (2FA not enabled)".into()))?;
        let wk = derive_wrap_key(seed, &salt)?;
        unwrap_key(blob, &wk)
    }

    /// Re-wrap the primary wrap under a new launch password (used to *change*
    /// the launch password without rekeying the DB). Requires the plaintext DB
    /// key, which the caller obtains by unwrapping with the old password first.
    pub fn change_primary(&mut self, db_key: &DerivedKey, new_launch_password: &str) -> Result<()> {
        let primary_salt = random_salt();
        let wk = derive_wrap_key(new_launch_password, &primary_salt)?;
        let primary_blob = wrap_key(db_key, &wk)?;
        self.primary_salt = primary_salt;
        self.primary_blob = primary_blob;
        Ok(())
    }

    /// Re-wrap a **new** DB key under the *existing* launch-password wrap key
    /// (used after a DB rekey, where the DB key changes but the launch password
    /// does not). Requires the launch password to re-derive the wrap key; keeps
    /// the existing primary_salt so the launch password still works at unlock.
    /// The recovery wrap is refreshed under the same (constant) seed.
    pub fn rewrap_for_new_db_key(
        &mut self,
        new_db_key: &DerivedKey,
        launch_password: &str,
    ) -> Result<()> {
        let wk = derive_wrap_key(launch_password, &self.primary_salt)?;
        self.primary_blob = wrap_key(new_db_key, &wk)?;
        if let Some(seed) = self.recovery_seed_b32.clone() {
            self.set_recovery(new_db_key, &seed)?;
        }
        Ok(())
    }

    /// Serialize to the on-disk text format (base64 lines).
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut s = String::new();
        s.push_str(KEY_BUNDLE_MAGIC);
        s.push('\n');
        s.push_str(&format!("primary_salt={}\n", B64.encode(self.primary_salt)));
        s.push_str(&format!("primary_blob={}\n", B64.encode(&self.primary_blob)));
        if let Some(seed) = &self.recovery_seed_b32 {
            s.push_str(&format!("recovery_seed={seed}\n"));
        }
        if let Some(salt) = self.recovery_salt {
            s.push_str(&format!("recovery_salt={}\n", B64.encode(salt)));
        }
        if let Some(blob) = &self.recovery_blob {
            s.push_str(&format!("recovery_blob={}\n", B64.encode(blob)));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, s)?;
        Ok(())
    }

    /// Load and parse a key bundle from disk. Errors if missing/corrupt.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| PhoenixError::Crypto(format!("read key bundle {}: {e}", path.display())))?;
        let mut lines = text.lines();
        let magic = lines
            .next()
            .ok_or_else(|| PhoenixError::Crypto("key bundle empty".into()))?;
        if magic != KEY_BUNDLE_MAGIC {
            return Err(PhoenixError::Crypto(format!(
                "key bundle has wrong header: {magic}"
            )));
        }
        let mut kv: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for line in lines {
            if let Some((k, v)) = line.split_once('=') {
                kv.insert(k.to_string(), v.to_string());
            }
        }
        let primary_salt_bytes = B64
            .decode(kv.get("primary_salt").ok_or_else(|| {
                PhoenixError::Crypto("key bundle missing primary_salt".into())
            })?)
            .map_err(|e| PhoenixError::Crypto(format!("primary_salt b64: {e}")))?;
        let primary_blob = B64
            .decode(kv.get("primary_blob").ok_or_else(|| {
                PhoenixError::Crypto("key bundle missing primary_blob".into())
            })?)
            .map_err(|e| PhoenixError::Crypto(format!("primary_blob b64: {e}")))?;
        if primary_salt_bytes.len() != SALT_LEN {
            return Err(PhoenixError::Crypto(format!(
                "primary_salt is {} bytes, expected {SALT_LEN}",
                primary_salt_bytes.len()
            )));
        }
        let mut primary_salt = [0u8; SALT_LEN];
        primary_salt.copy_from_slice(&primary_salt_bytes);

        let recovery_seed_b32 = kv.get("recovery_seed").cloned();
        let (recovery_salt, recovery_blob) = match (kv.get("recovery_salt"), kv.get("recovery_blob"))
        {
            (Some(s), Some(b)) => {
                let salt_bytes = B64
                    .decode(s)
                    .map_err(|e| PhoenixError::Crypto(format!("recovery_salt b64: {e}")))?;
                if salt_bytes.len() != SALT_LEN {
                    return Err(PhoenixError::Crypto(format!(
                        "recovery_salt is {} bytes, expected {SALT_LEN}",
                        salt_bytes.len()
                    )));
                }
                let mut salt = [0u8; SALT_LEN];
                salt.copy_from_slice(&salt_bytes);
                let blob = B64
                    .decode(b)
                    .map_err(|e| PhoenixError::Crypto(format!("recovery_blob b64: {e}")))?;
                (Some(salt), Some(blob))
            }
            _ => (None, None),
        };

        Ok(Self {
            primary_salt,
            primary_blob,
            recovery_seed_b32,
            recovery_salt,
            recovery_blob,
        })
    }
}
