//! RFC 6238 TOTP (time-based one-time passwords) for optional 2FA.
//!
//! Compatible with Google Authenticator, Proton Pass, 2FAS, Aegis, and any
//! standard authenticator app — the secret is exchanged as a base32 string and
//! an `otpauth://` provisioning URI (rendered to a QR code in the UI).
//!
//! ## How it's used
//! At unlock, the user's current 6-digit code is appended to their passphrase
//! before Argon2id key derivation (see [`super::derive_key`]). So a valid code
//! is required to derive the correct DB key.
//!
//! ## Honest security caveat
//! The TOTP seed is stored *inside* the encrypted database (it must be, so we
//! can verify codes). That means anyone with the DB file has the seed too.
//! Consequently this is a *second knowledge factor derived from a time code*,
//! not a true "something you have" factor. Its real value is defeating offline
//! passphrase-guessing: each brute-force attempt must also try ~1M codes, and
//! the attacker can't tell which attempts are "close". Use a key-file (future
//! work) if you need a genuine second factor on separate media.

use base32::Alphabet;
use serde::Serialize;
use totp_rs::{Algorithm, Secret, TOTP};

use crate::error::{PhoenixError, Result};

/// TOTP step in seconds (RFC 6238 default).
pub const STEP: u64 = 30;
/// Number of digits in the code (RFC 6238 default).
pub const DIGITS: usize = 6;
/// Allowed clock skew (in steps, each side). 1 = ±30s tolerance.
pub const SKEW: u8 = 1;

/// A configured TOTP, plus the base32 secret string used to add it to an
/// authenticator app and the `otpauth://` URI for QR rendering.
#[derive(Debug, Clone, Serialize)]
pub struct TotpSetup {
    /// The raw secret as a base32 string (no padding), e.g. `"JBSWY3DPEHPK3PXP"`.
    pub secret_b32: String,
    /// `otpauth://totp/...` URI; the frontend encodes it into a QR code.
    pub otpauth_url: String,
}

/// A generated-but-unpersisted TOTP (returned by `setup_totp` for confirmation).
pub struct TotpInstance {
    pub totp: TOTP,
    pub secret_b32: String,
}

/// Generate a fresh random secret and build a TOTP instance for `account`.
pub fn generate(account: &str) -> Result<TotpInstance> {
    // Secret::generate_secret() yields 20 random bytes (160 bits, RFC 4226).
    let secret = Secret::generate_secret();
    let secret_bytes = secret_bytes(&secret)?;
    let secret_b32 = encode_b32(&secret_bytes);
    let totp = build(&secret_bytes, account)?;
    Ok(TotpInstance { totp, secret_b32 })
}

/// Reconstruct a TOTP instance from a stored base32 secret.
pub fn from_secret(secret_b32: &str, account: &str) -> Result<TOTP> {
    let bytes = decode_b32(secret_b32)?;
    build(&bytes, account)
}

fn build(secret_bytes: &[u8], account: &str) -> Result<TOTP> {
    TOTP::new(
        Algorithm::SHA1,
        DIGITS,
        SKEW,
        STEP,
        secret_bytes.to_vec(),
        Some("PhoenixAgent".to_string()),
        account.to_string(),
    )
    .map_err(|e| PhoenixError::Crypto(format!("totp build: {e}")))
}

/// The current 6-digit code (for the live clock).
pub fn current_code(totp: &TOTP) -> String {
    totp.generate_current().unwrap_or_default()
}

/// Verify a user-entered code against the current time window, using the skew
/// configured at build time (±1 step). Constant-time comparison is handled
/// internally by `totp-rs`.
pub fn verify(totp: &TOTP, code: &str) -> bool {
    let code = code.trim();
    if code.len() != DIGITS || !code.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    totp.check_current(code).unwrap_or(false)
}

/// Rebuild a TOTP instance from a base32 seed (with a placeholder account, since
/// the account label doesn't affect code generation) and verify a typed code.
/// Used by the key-bundle recovery path, which stores only the seed (not the
/// account label). Errors on a bad seed; returns Ok(()) on match, Err on miss.
pub fn verify_raw(secret_b32: &str, code: &str) -> Result<()> {
    let totp = from_secret(secret_b32, "recovery")?;
    if verify(&totp, code) {
        Ok(())
    } else {
        Err(PhoenixError::Crypto("recovery code does not match".into()))
    }
}

/// Validate that a base32 string decodes to a usable secret. Returns the
/// decoded byte length on success.
pub fn validate_secret(secret_b32: &str) -> Result<usize> {
    let bytes = decode_b32(secret_b32)?;
    if bytes.len() < 10 {
        return Err(PhoenixError::Crypto(format!(
            "totp secret too short ({} bytes; need ≥10)",
            bytes.len()
        )));
    }
    Ok(bytes.len())
}

fn encode_b32(bytes: &[u8]) -> String {
    base32::encode(Alphabet::Rfc4648 { padding: false }, bytes)
}

fn decode_b32(s: &str) -> Result<Vec<u8>> {
    base32::decode(Alphabet::Rfc4648 { padding: false }, s)
        .ok_or_else(|| PhoenixError::Crypto("invalid base32 secret".into()))
}

/// Pull the raw bytes out of a `Secret` (Raw or Encoded).
fn secret_bytes(secret: &Secret) -> Result<Vec<u8>> {
    match secret {
        Secret::Raw(b) => Ok(b.clone()),
        Secret::Encoded(s) => decode_b32(s),
    }
}

impl TotpInstance {
    /// Render the provisioning info the UI needs (secret + otpauth URL).
    pub fn to_setup(&self) -> TotpSetup {
        TotpSetup {
            secret_b32: self.secret_b32.clone(),
            otpauth_url: self.totp.get_url(),
        }
    }
}
