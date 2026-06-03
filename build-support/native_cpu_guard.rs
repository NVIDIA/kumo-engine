use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const ALLOW_NATIVE_CPU_ENV: &str = "DISKGRAPH_ALLOW_NATIVE_CPU_BUILD";

pub fn reject_native_cpu_build(artifact: &str) {
    println!("cargo:rerun-if-env-changed={ALLOW_NATIVE_CPU_ENV}");
    for name in rustflag_env_names() {
        println!("cargo:rerun-if-env-changed={name}");
    }

    if env::var_os(ALLOW_NATIVE_CPU_ENV).is_some() {
        println!("cargo:warning={ALLOW_NATIVE_CPU_ENV} is set; allowing target-cpu=native");
        return;
    }

    let mut offenders = Vec::new();
    for name in rustflag_env_names() {
        let Ok(value) = env::var(&name) else {
            continue;
        };
        if contains_native_cpu_flag(&value.replace('\x1f', " ")) {
            offenders.push(format!("{name} contains target-cpu=native"));
        }
    }

    for path in cargo_config_paths() {
        println!("cargo:rerun-if-changed={}", path.display());
        let Ok(contents) = fs::read_to_string(&path) else {
            continue;
        };
        if cargo_config_contains_native_cpu(&contents) {
            offenders.push(format!("{} contains target-cpu=native", path.display()));
        }
    }

    if offenders.is_empty() {
        return;
    }

    panic!(
        "Refusing to build {artifact} with target-cpu=native ({offenders}). \
         Diskgraph wheels ship native code and the standalone ingest binary to \
         Kumo workers; host-specific CPU instructions can crash there as SIGILL. \
         Use portable target features from .cargo/config.toml, or set \
         {ALLOW_NATIVE_CPU_ENV}=1 only for a private same-machine build.",
        offenders = offenders.join("; "),
    );
}

fn rustflag_env_names() -> Vec<String> {
    let mut names = vec![
        "RUSTFLAGS".to_owned(),
        "CARGO_ENCODED_RUSTFLAGS".to_owned(),
        "CARGO_BUILD_RUSTFLAGS".to_owned(),
    ];

    if let Ok(target) = env::var("TARGET") {
        let normalized: String = target
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() {
                    ch.to_ascii_uppercase()
                } else {
                    '_'
                }
            })
            .collect();
        names.push(format!("CARGO_TARGET_{normalized}_RUSTFLAGS"));
    }

    names
}

fn cargo_config_paths() -> Vec<PathBuf> {
    let mut paths = BTreeSet::new();

    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let mut current = Some(PathBuf::from(manifest_dir));
        while let Some(dir) = current {
            push_cargo_config_candidates(&mut paths, &dir.join(".cargo"));
            current = dir.parent().map(Path::to_path_buf);
        }
    }

    if let Some(cargo_home) = env::var_os("CARGO_HOME") {
        push_cargo_config_candidates(&mut paths, Path::new(&cargo_home));
    } else if let Some(home) = env::var_os("HOME") {
        push_cargo_config_candidates(&mut paths, &Path::new(&home).join(".cargo"));
    }

    paths.into_iter().collect()
}

fn push_cargo_config_candidates(paths: &mut BTreeSet<PathBuf>, cargo_dir: &Path) {
    for name in ["config.toml", "config"] {
        let path = cargo_dir.join(name);
        if path.exists() {
            paths.insert(path);
        }
    }
}

fn cargo_config_contains_native_cpu(contents: &str) -> bool {
    contains_native_cpu_flag(&strip_toml_comments(contents))
}

fn contains_native_cpu_flag(flags: &str) -> bool {
    let compact: String = flags
        .chars()
        .filter(|ch| !ch.is_whitespace() && *ch != '"' && *ch != '\'')
        .collect();
    compact.contains("target-cpu=native")
}

fn strip_toml_comments(contents: &str) -> String {
    let mut out = String::with_capacity(contents.len());
    let mut in_string = false;
    let mut string_quote = '\0';
    let mut escaped = false;
    let mut in_comment = false;

    for ch in contents.chars() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
                out.push(ch);
            }
            continue;
        }

        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == string_quote {
                in_string = false;
            }
            continue;
        }

        if ch == '"' || ch == '\'' {
            in_string = true;
            string_quote = ch;
            out.push(ch);
        } else if ch == '#' {
            in_comment = true;
        } else {
            out.push(ch);
        }
    }

    out
}
