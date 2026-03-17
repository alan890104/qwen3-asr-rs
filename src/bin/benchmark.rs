/// Benchmark binary for qwen3-asr.
///
/// Usage:
///   benchmark [--model-dir dir] [--runs N] [--audio-dir dir]
///
/// If --model-dir is not given, tries "models/" as the safetensors dir.
/// --runs N: number of transcription iterations per audio file (default 1).
///
/// Reports:
///   - Load time
///   - Per-file transcription time (mean / min / max over N runs)
///   - Wall-clock throughput (audio-seconds transcribed per second)

use anyhow::Result;
use qwen3_asr::{AsrInference, TranscribeOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ─── CLI ──────────────────────────────────────────────────────────────────────

struct Cli {
    model_dir: Option<PathBuf>,
    #[cfg(feature = "hub")]
    model_id: Option<String>,
    #[cfg(feature = "hub")]
    cache_dir: PathBuf,
    audio_dir: PathBuf,
    runs: usize,
    label: String,
}

fn parse_args() -> Cli {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<PathBuf> = None;
    #[cfg(feature = "hub")]
    let mut model_id: Option<String> = None;
    #[cfg(feature = "hub")]
    let mut cache_dir = PathBuf::from("models");
    let mut audio_dir = PathBuf::from("tests/fixtures/audio");
    let mut runs: usize = 1;
    let mut label = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                model_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--model-id" => {
                #[cfg(feature = "hub")]
                { model_id = Some(args[i + 1].clone()); }
                i += 2;
            }
            "--cache-dir" => {
                #[cfg(feature = "hub")]
                { cache_dir = PathBuf::from(&args[i + 1]); }
                i += 2;
            }
            "--audio-dir" => {
                audio_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--runs" => {
                runs = args[i + 1].parse().unwrap_or(1);
                i += 2;
            }
            "--label" => {
                label = args[i + 1].clone();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Auto-label from source if not supplied.
    if label.is_empty() {
        #[cfg(feature = "hub")]
        if let Some(ref id) = model_id {
            label = format!("hub({})", id);
        }
        if label.is_empty() {
            label = if let Some(ref d) = model_dir {
                format!("safetensors({})", d.display())
            } else {
                "safetensors(models/)".to_string()
            };
        }
    }

    Cli {
        model_dir,
        #[cfg(feature = "hub")]
        model_id,
        #[cfg(feature = "hub")]
        cache_dir,
        audio_dir,
        runs,
        label,
    }
}

// ─── Stats ────────────────────────────────────────────────────────────────────

fn stats(durations: &[Duration]) -> (Duration, Duration, Duration) {
    let min = *durations.iter().min().unwrap();
    let max = *durations.iter().max().unwrap();
    let mean = durations.iter().sum::<Duration>() / durations.len() as u32;
    (mean, min, max)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

// ─── Memory metrics (macOS) ───────────────────────────────────────────────────

/// Peak RSS in MiB from getrusage(RUSAGE_SELF).
/// This is the high-water mark since process start — it never decreases.
/// On macOS ru_maxrss is bytes; on Linux it is kibibytes.
fn peak_rss_mib() -> f64 {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        #[repr(C)]
        struct Rusage {
            ru_utime: [i64; 2],
            ru_stime: [i64; 2],
            ru_maxrss: i64,
            _pad: [i64; 13],
        }
        let mut u = Rusage { ru_utime: [0; 2], ru_stime: [0; 2], ru_maxrss: 0, _pad: [0; 13] };
        unsafe { getrusage(0, &mut u) };
        u.ru_maxrss as f64 / 1024.0 / 1024.0
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb) = line.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()) {
                        return kb / 1024.0;
                    }
                }
            }
        }
        0.0
    }
}

/// Physical memory footprint in MiB from task_info TASK_VM_INFO (macOS only).
///
/// Unlike getrusage, `phys_footprint` reflects the *current* physical pages
/// owned by the process (including Metal/GPU buffers in unified memory) and
/// can decrease after large allocations are freed.  This is the most accurate
/// indicator of how much RAM the model actually occupies after loading.
fn phys_footprint_mib() -> f64 {
    #[cfg(target_os = "macos")]
    {
        // task_vm_info (flavor 22) struct layout at 64-bit offsets:
        //   byte   0: virtual_size        u64
        //   byte   8: region_count        i32
        //   byte  12: page_size           i32
        //   byte  16: resident_size       u64
        //   byte  24: resident_size_peak  u64
        //   byte  32: device              u64
        //   byte  40: device_peak         u64
        //   byte  48: internal            u64
        //   byte  56: internal_peak       u64
        //   byte  64: external            u64
        //   byte  72: external_peak       u64
        //   byte  80: reusable            u64
        //   byte  88: reusable_peak       u64
        //   byte  96: purgeable_volatile_pmap       u64
        //   byte 104: purgeable_volatile_resident   u64
        //   byte 112: purgeable_volatile_virtual    u64
        //   byte 120: compressed          u64
        //   byte 128: compressed_peak     u64  ← u64 index 16
        //   byte 136: compressed_lifetime u64  ← u64 index 17
        //   byte 144: phys_footprint      u64  ← u64 index 18
        extern "C" {
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut u64,
                task_info_outCnt: *mut u32,
            ) -> i32;
            fn mach_task_self() -> u32;
        }
        const TASK_VM_INFO: u32 = 22;
        // count is in natural_t (i32-sized) units; [u64; 40] = 320 bytes = 80 i32 units
        let mut buf = [0u64; 40];
        let mut count: u32 = (buf.len() * 2) as u32;
        let ret = unsafe { task_info(mach_task_self(), TASK_VM_INFO, buf.as_mut_ptr(), &mut count) };
        if ret != 0 { return 0.0; }
        // phys_footprint is at byte offset 144 = u64 index 18
        let footprint_bytes = buf[18];
        footprint_bytes as f64 / 1024.0 / 1024.0
    }
    #[cfg(not(target_os = "macos"))]
    {
        0.0
    }
}

// ─── Audio duration ───────────────────────────────────────────────────────────

fn wav_duration_secs(path: &Path) -> f64 {
    let reader = match hound::WavReader::open(path) {
        Ok(r) => r,
        Err(_) => return 0.0,
    };
    let spec = reader.spec();
    let num_samples = reader.duration();
    num_samples as f64 / spec.sample_rate as f64
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = parse_args();

    let device = qwen3_asr::best_device();

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  qwen3-asr benchmark — {}", cli.label);
    println!("═══════════════════════════════════════════════════════════");
    println!("  Device : {:?}", device);
    println!("  Runs   : {}", cli.runs);
    println!();

    // ── Load model ────────────────────────────────────────────────────────────
    let t_load = Instant::now();
    #[cfg(feature = "hub")]
    let engine = if let Some(ref id) = cli.model_id {
        AsrInference::from_pretrained(id, &cli.cache_dir, device)?
    } else {
        let dir = cli.model_dir.as_deref().unwrap_or_else(|| Path::new("models"));
        AsrInference::load(dir, device)?
    };
    #[cfg(not(feature = "hub"))]
    let engine = {
        let dir = cli.model_dir.as_deref().unwrap_or_else(|| Path::new("models"));
        AsrInference::load(dir, device)?
    };
    let load_time = t_load.elapsed();
    let rss_after_load = peak_rss_mib();
    let footprint_after_load = phys_footprint_mib();

    println!("  Load time   : {:.1} ms", ms(load_time));
    println!("  Peak RSS    : {:.0} MiB  (getrusage high-water mark — never decreases)", rss_after_load);
    println!("  Phys footpt : {:.0} MiB  (task_info current physical memory including Metal GPU)", footprint_after_load);
    println!();

    // ── Audio files to benchmark ──────────────────────────────────────────────
    let audio_files: Vec<PathBuf> = {
        let mut v: Vec<_> = std::fs::read_dir(&cli.audio_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
            .collect();
        v.sort();
        v
    };

    if audio_files.is_empty() {
        eprintln!("No .wav files found in {}", cli.audio_dir.display());
        return Ok(());
    }

    // ── Transcription benchmark ───────────────────────────────────────────────
    let mut total_audio_secs: f64 = 0.0;
    let mut total_infer_secs: f64 = 0.0;

    println!("{:<20}  {:>8}  {:>8}  {:>8}  {:>8}  {}",
        "File", "Mean ms", "Min ms", "Max ms", "RTF", "Text");
    println!("{}", "─".repeat(100));

    for wav in &audio_files {
        let wav_str = wav.to_str().unwrap_or_default();
        let audio_dur = wav_duration_secs(wav);

        let mut times: Vec<Duration> = Vec::with_capacity(cli.runs);
        let mut last_text = String::new();

        for _ in 0..cli.runs {
            let t = Instant::now();
            let result = engine.transcribe(wav_str, TranscribeOptions::default())?;
            times.push(t.elapsed());
            last_text = result.text.trim().to_string();
        }

        let (mean, min, max) = stats(&times);
        let rtf = if audio_dur > 0.0 {
            mean.as_secs_f64() / audio_dur
        } else {
            0.0
        };

        // Truncate text display (char-safe to handle multibyte UTF-8).
        let text_display: String = {
            let chars: Vec<char> = last_text.chars().collect();
            if chars.len() > 40 {
                format!("{}…", chars[..40].iter().collect::<String>())
            } else {
                last_text.clone()
            }
        };

        let short_name = wav.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(wav_str);

        println!("{:<20}  {:>8.1}  {:>8.1}  {:>8.1}  {:>8.3}  {}",
            short_name, ms(mean), ms(min), ms(max), rtf, text_display);

        total_audio_secs += audio_dur;
        total_infer_secs += mean.as_secs_f64();
    }

    println!("{}", "─".repeat(100));
    let overall_rtf = if total_audio_secs > 0.0 {
        total_infer_secs / total_audio_secs
    } else {
        0.0
    };
    println!("{:<20}  {:>8.1}  {:>8}  {:>8}  {:>8.3}",
        "TOTAL/AVG", ms(Duration::from_secs_f64(total_infer_secs)),
        "", "", overall_rtf);
    println!();
    println!("  RTF = inference_time / audio_duration  (lower is better)");
    println!("  Peak RSS (getrusage high-water)       : {:.0} MiB", rss_after_load);
    println!("  Phys footprint (task_info, post-load) : {:.0} MiB", footprint_after_load);
    println!();

    Ok(())
}
