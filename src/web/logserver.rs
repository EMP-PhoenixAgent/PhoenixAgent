//! A one-off local HTTP server that gives LogsExplorer its "full power"
//! mode. A double-clicked `LogsExplorer.html` is read-only (browsers cannot
//! delete files from `file://`); when opened from the app (Settings → Logs →
//! *Open LogsExplorer*), this server serves the same page plus a `/delete`
//! endpoint so the cleanup buttons actually work. It binds to a random
//! loopback port, serves nothing outside `Logs/`, and dies with the process.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use crate::logsys;

static URL: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Start (or return the already-running) explorer server and open it in the
/// default browser. Returns the URL.
pub fn start(logs_dir: &PathBuf) -> std::io::Result<String> {
    if let Some(url) = URL.lock().unwrap().clone() {
        return Ok(url);
    }
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    let dir = logs_dir.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(stream) = conn else { continue };
            let dir = dir.clone();
            std::thread::spawn(move || handle(stream, dir));
        }
    });
    let url = format!("http://127.0.0.1:{port}/");
    *URL.lock().unwrap() = Some(url.clone());
    #[cfg(windows)]
    let _ = std::process::Command::new("cmd")
        .args(["/c", "start", "", &url])
        .spawn();
    #[cfg(not(windows))]
    {
        let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
    }
    Ok(url)
}

/// Handle exactly one HTTP/1.1 request per connection (Connection: close).
fn handle(mut stream: TcpStream, dir: PathBuf) {
    let Some((method, path, body)) = read_request(&mut stream) else { return };
    let (status, ctype, body) = route(&method, &path, &body, &dir);
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&body);
    let _ = stream.flush();
}

/// Read until the end of headers, then the Content-Length body. Caps at
/// 1 MB — delete payloads are a few dozen bytes.
fn read_request(stream: &mut TcpStream) -> Option<(String, String, Vec<u8>)> {
    let mut buf: Vec<u8> = Vec::with_capacity(2048);
    let mut chunk = [0u8; 2048];
    loop {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_headers_end(&buf) {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            let len: usize = head
                .lines()
                .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
            let total = pos + 4 + len;
            if buf.len() >= total.min(1024 * 1024) {
                let mut lines = head.lines();
                let request = lines.next()?.to_string();
                let mut parts = request.split_whitespace();
                let method = parts.next()?.to_string();
                let path = parts.next()?.to_string();
                let body = buf[pos + 4..total].to_vec();
                return Some((method, path, body));
            }
        }
    }
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn route(method: &str, path: &str, body: &[u8], dir: &PathBuf) -> (&'static str, &'static str, Vec<u8>) {
    match (method, path) {
        ("GET", "/") => ("200 OK", "text/html; charset=utf-8", logsys::explorer_html().as_bytes().to_vec()),
        ("GET", "/data.js") => {
            let catalog = serde_json::to_string(&logsys::build_catalog(dir)).unwrap_or_else(|_| "{}".into());
            (
                "200 OK",
                "application/javascript; charset=utf-8",
                format!("window.PA_SERVED=true;window.PA_LOGS = {catalog};\n").into_bytes(),
            )
        }
        ("POST", "/delete") => {
            let removed = serde_json::from_slice::<serde_json::Value>(body)
                .ok()
                .and_then(|v| {
                    let files = v
                        .get("files")
                        .and_then(|f| f.as_array())
                        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect());
                    match (v.get("scope").and_then(|s| s.as_str()), files) {
                        (Some("clean"), _) => Some(logsys::DeleteScope::Clean),
                        (Some("all"), _) => Some(logsys::DeleteScope::All),
                        (_, files) => files.map(logsys::DeleteScope::Files),
                    }
                })
                .map(|scope| logsys::delete(dir, scope))
                .unwrap_or(0);
            (
                "200 OK",
                "application/json",
                format!("{{\"ok\":true,\"removed\":{removed}}}").into_bytes(),
            )
        }
        _ => ("404 Not Found", "text/plain", b"not found".to_vec()),
    }
}
