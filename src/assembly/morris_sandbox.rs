use std::os::unix::process::ExitStatusExt;
use std::io::BufRead;
use std::process::{Command, Output};
use std::time::{Duration, Instant};
use std::path::PathBuf;
use crate::layers::syntax_layer::ViolationKind;

#[derive(Clone, Debug)]
pub struct SandboxOutcome {
    pub compiles: bool,
    pub runs: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub violations: Vec<ViolationKind>,
    pub execution_time_us: u64,
    pub anomaly_triggered: bool,
}

impl SandboxOutcome {
    pub fn failed(reason: &str) -> Self {
        SandboxOutcome {
            compiles: false,
            runs: false,
            exit_code: None,
            stdout: String::new(),
            stderr: reason.to_string(),
            violations: Vec::new(),
            execution_time_us: 0,
            anomaly_triggered: false,
        }
    }
}

pub struct MorrisSandbox {
    temp_dir: PathBuf,
    compile_timeout: Duration,
    exec_timeout: Duration,
}

impl MorrisSandbox {
    pub fn new() -> Self {
        let temp_dir = std::env::temp_dir().join("fuga_morris");
        std::fs::create_dir_all(&temp_dir).ok();
        MorrisSandbox {
            temp_dir,
            compile_timeout: Duration::from_secs(60),
            exec_timeout: Duration::from_secs(10),
        }
    }

    pub fn with_timeouts(compile_secs: u64, exec_secs: u64) -> Self {
        let temp_dir = std::env::temp_dir().join("fuga_morris");
        std::fs::create_dir_all(&temp_dir).ok();
        MorrisSandbox {
            temp_dir,
            compile_timeout: Duration::from_secs(compile_secs),
            exec_timeout: Duration::from_secs(exec_secs),
        }
    }

    pub fn evaluate_code(&self, code: &str, file_name: &str) -> SandboxOutcome {
        let file_path = self.temp_dir.join(file_name);
        let exe_name = file_name.trim_end_matches(".rs");
        let exe_path = self.temp_dir.join(exe_name);

        if let Err(e) = std::fs::write(&file_path, code) {
            return SandboxOutcome::failed(&format!("write error: {}", e));
        }

        let compile_out = self.compile(&file_path, &exe_path, exe_name);
        if !compile_out.status.success() {
            let stderr = String::from_utf8_lossy(&compile_out.stderr);
            let violations = scan_violations(&stderr);
            let triggered = !violations.is_empty();
            return SandboxOutcome {
                compiles: false,
                runs: false,
                exit_code: compile_out.status.code(),
                stdout: String::from_utf8_lossy(&compile_out.stdout).to_string(),
                stderr: stderr.to_string(),
                violations,
                execution_time_us: 0,
                anomaly_triggered: triggered,
            };
        }

        let start = Instant::now();
        let run_out = match self.run_binary(&exe_path) {
            Some(output) => output,
            None => {
                return SandboxOutcome {
                    compiles: true,
                    runs: false,
                    exit_code: None,
                    stdout: String::new(),
                    stderr: "TIMEOUT".into(),
                    violations: vec![ViolationKind::InfiniteLoop],
                    execution_time_us: self.exec_timeout.as_micros() as u64,
                    anomaly_triggered: true,
                };
            }
        };
        let elapsed = start.elapsed().as_micros() as u64;

        let stdout = String::from_utf8_lossy(&run_out.stdout).to_string();
        let stderr = String::from_utf8_lossy(&run_out.stderr).to_string();
        let mut violations = scan_violations(&stdout);
        violations.extend(scan_violations(&stderr));
        let triggered = !violations.is_empty();

        SandboxOutcome {
            compiles: true,
            runs: run_out.status.success(),
            exit_code: run_out.status.code(),
            stdout,
            stderr,
            violations,
            execution_time_us: elapsed,
            anomaly_triggered: triggered,
        }
    }

    fn compile(&self, file_path: &PathBuf, exe_path: &PathBuf, _crate_name: &str) -> Output {
        let mut cmd = Command::new("rustc");
        cmd.arg("--edition")
            .arg("2021")
            .arg(file_path)
            .arg("-o")
            .arg(exe_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        cmd.output().unwrap_or_else(|_| {
            Output {
                status: std::process::ExitStatus::from_raw(1),
                stdout: Vec::new(),
                stderr: b"rustc not found".to_vec(),
            }
        })
    }

    fn run_binary(&self, exe_path: &PathBuf) -> Option<Output> {
        let mut child = Command::new(exe_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .ok()?;

        let now = Instant::now();
        loop {
            if now.elapsed() >= self.exec_timeout {
                let _ = std::process::Command::new("kill")
                    .arg(child.id().to_string())
                    .output();
                return None;
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    let stdout = std::io::BufReader::new(child.stdout.take()?)
                        .lines()
                        .collect::<Result<Vec<_>, _>>()
                        .ok()?
                        .join("\n");
                    let stderr = std::io::BufReader::new(child.stderr.take()?)
                        .lines()
                        .collect::<Result<Vec<_>, _>>()
                        .ok()?
                        .join("\n");
                    return Some(Output {
                        status,
                        stdout: stdout.into_bytes(),
                        stderr: stderr.into_bytes(),
                    });
                }
                Ok(None) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => return None,
            }
        }
    }
}

fn scan_violations(text: &str) -> Vec<ViolationKind> {
    let mut v = Vec::new();
    if text.contains("panic!") || text.contains("panicked") {
        v.push(ViolationKind::InfiniteLoop);
    }
    if text.contains("index out of bounds") || text.contains("out of bounds") {
        v.push(ViolationKind::ArrayIndexOutOfBounds);
    }
    if text.contains("segmentation fault") || text.contains("SIGSEGV") {
        v.push(ViolationKind::NullPointerDeref);
    }
    if text.contains("division by zero") || text.contains("DivideByZero") {
        v.push(ViolationKind::DivisionByZero);
    }
    if text.contains("TIMEOUT") || text.contains("timed out") {
        v.push(ViolationKind::InfiniteLoop);
    }
    if text.contains("overflow") || text.contains("Overflow") {
        v.push(ViolationKind::IntegerOverflow);
    }
    if text.contains("unsafe") || text.contains("Unsafe") {
        v.push(ViolationKind::UnsafeBlock);
    }
    v
}
