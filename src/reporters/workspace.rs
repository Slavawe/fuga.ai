use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Режим сканирования
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScanMode {
    /// Сканировать один файл
    SingleFile,
    /// Рекурсивное сканирование директории
    Recursive,
    /// Сканировать Cargo workspace
    Workspace,
}

/// Расширения поддерживаемых языков
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "rs", "c", "h", "cpp", "cc", "cxx", "hpp", "hh", "go", "py", "ts", "tsx", "js", "jsx",
];

fn is_supported_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|s| s.to_str())
            .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext))
            .unwrap_or(false)
}

/// Сканер workspace и рекурсивных директорий
pub struct WorkspaceScanner;

impl WorkspaceScanner {
    pub fn new() -> Self {
        Self
    }

    /// Сканирует путь в соответствии с режимом и возвращает список файлов
    pub fn scan(&self, path: &str, mode: ScanMode) -> Result<Vec<PathBuf>, String> {
        let path_buf = PathBuf::from(path);

        match mode {
            ScanMode::SingleFile => {
                if !path_buf.exists() {
                    return Err(format!("File not found: {}", path));
                }
                if !path_buf.is_file() {
                    return Err(format!("Not a file: {}", path));
                }
                if !is_supported_file(&path_buf) {
                    return Err(format!("Unsupported file extension: {}", path));
                }
                Ok(vec![path_buf])
            }
            ScanMode::Recursive => self.scan_recursive(&path_buf),
            ScanMode::Workspace => self.scan_workspace(&path_buf),
        }
    }

    fn scan_recursive(&self, path: &Path) -> Result<Vec<PathBuf>, String> {
        if !path.exists() {
            return Err(format!("Path not found: {}", path.display()));
        }

        let mut files = Vec::new();

        for entry in WalkDir::new(path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if is_supported_file(path) {
                // Пропускаем target/ и .git/
                if path
                    .components()
                    .any(|c| c.as_os_str() == "target" || c.as_os_str() == ".git")
                {
                    continue;
                }
                files.push(path.to_path_buf());
            }
        }

        if files.is_empty() {
            return Err(format!(
                "No supported source files found in {}",
                path.display()
            ));
        }

        Ok(files)
    }

    fn scan_workspace(&self, path: &Path) -> Result<Vec<PathBuf>, String> {
        // Ищем Cargo.toml
        let cargo_toml = if path.is_file() && path.file_name() == Some("Cargo.toml".as_ref()) {
            path.to_path_buf()
        } else if path.is_dir() {
            path.join("Cargo.toml")
        } else {
            return Err("Expected directory or Cargo.toml file".to_string());
        };

        if !cargo_toml.exists() {
            return Err(format!("Cargo.toml not found at {}", cargo_toml.display()));
        }

        // Парсим Cargo.toml
        let content = fs::read_to_string(&cargo_toml)
            .map_err(|e| format!("Failed to read Cargo.toml: {}", e))?;

        let manifest: toml::Value =
            toml::from_str(&content).map_err(|e| format!("Failed to parse Cargo.toml: {}", e))?;

        let workspace_root = cargo_toml.parent().ok_or("Invalid Cargo.toml path")?;

        // Проверяем, есть ли [workspace]
        if let Some(workspace) = manifest.get("workspace") {
            // Workspace with members
            let members = workspace
                .get("members")
                .and_then(|m| m.as_array())
                .ok_or("workspace.members not found or not an array")?;

            let mut all_files = Vec::new();

            for member in members {
                let member_path = member.as_str().ok_or("Invalid member path")?;
                let member_dir = workspace_root.join(member_path);

                // Сканируем src/ каждого члена
                let src_dir = member_dir.join("src");
                if src_dir.exists() {
                    match self.scan_recursive(&src_dir) {
                        Ok(files) => all_files.extend(files),
                        Err(_) => continue,
                    }
                }
            }

            if all_files.is_empty() {
                return Err("No supported files found in workspace members".to_string());
            }

            Ok(all_files)
        } else {
            // Single package — сканируем его src/
            let src_dir = workspace_root.join("src");
            if !src_dir.exists() {
                return Err(format!("src/ directory not found at {}", src_dir.display()));
            }
            self.scan_recursive(&src_dir)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_single_file() {
        let scanner = WorkspaceScanner::new();
        // Assuming we're in the fuga project directory
        let result = scanner.scan("src/main.rs", ScanMode::SingleFile);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }
}
