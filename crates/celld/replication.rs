// Copyright 2026 Deno Land Inc. Apache-2.0 license.

//! Shared replication primitives for the in-process LTX backend
//! ([`crate::ltx_repl::LtxRepl`]). Each cell db lives at
//! `<watch>/<cell>/ltx/e<epoch>/db.sqlite` and replicates to
//! `cells/<cell>/ltx/e<epoch>/` in the bucket — epoch-in-prefix is the
//! data-path fence: a stale owner writes a dead prefix.
use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// The first retry delay for a cell-release operation.
///
/// The actor uses this delay when a host keeps a cell. The snapshot publisher
/// uses it as the first step of a linear backoff after the L0 proof passes.
pub(crate) const CELL_RELEASE_RETRY_BASE_MS: u64 = 50;

/// The remote artifact that a successor can use after an eviction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionRestoreArtifact {
    /// A full L9 snapshot covers the closed database.
    Snapshot,
    /// The proven additive L0 chain remains the restore path.
    L0Chain,
}

/// The one bit that lets a revocable eviction give the cell back.
///
/// The core sets it when a request arrives for a cell in `Phase::Cleaning`,
/// after the runtime is gone and before the eviction has taken the database.
/// `LtxRepl::evict` reads it at the points where it can still stop, which are
/// the points before the handoff snapshot becomes visible.
#[derive(Default)]
pub struct EvictionAbandon(std::sync::atomic::AtomicBool);

impl EvictionAbandon {
    /// Ask the eviction to stop. Repeating the request changes nothing, so a
    /// second waiting request costs no more than the first.
    pub fn request(&self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn requested(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// The eviction stopped and left the cell resident at its own epoch.
///
/// This is an error so that it travels the path a failed release already
/// takes, and it is a distinct type so that the shell tells it apart from a
/// release that should be retried. Retrying an abandoned eviction would evict
/// the cell the abandonment just saved.
#[derive(Debug)]
pub struct EvictionAbandoned {
    pub cell: String,
    pub epoch: u64,
}

impl std::fmt::Display for EvictionAbandoned {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "abandoned the eviction of {} epoch {}; a request arrived before \
             the handoff snapshot became visible",
            self.cell, self.epoch
        )
    }
}

impl std::error::Error for EvictionAbandoned {}

/// Outcome of a blocking replication wait on one cell db.
pub enum SyncWait {
    /// The latest local commit is in the bucket.
    Durable,
    /// The replicator does not track this cell; the caller decides its
    /// fallback.
    Unsupported,
    /// The wait failed or timed out.
    Failed,
}

pub struct RestoredSnapshot {
    pub epoch: u64,
    path: PathBuf,
    directory: PathBuf,
    filesystem: Arc<dyn celld_ltx::FileSystem>,
}

impl RestoredSnapshot {
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Construct a snapshot whose `directory` is removed on drop, handing the
    /// caller an inspection copy with RAII cleanup.
    pub(crate) fn new(
        epoch: u64,
        path: PathBuf,
        directory: PathBuf,
        filesystem: Arc<dyn celld_ltx::FileSystem>,
    ) -> Self {
        Self {
            epoch,
            path,
            directory,
            filesystem,
        }
    }
}

impl Drop for RestoredSnapshot {
    fn drop(&mut self) {
        let _ = self.filesystem.remove_dir_all(&self.directory);
    }
}

#[derive(Clone)]
pub struct StorageCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

pub struct ActivationOptions<'a> {
    pub cell: &'a str,
    pub epoch: u64,
    /// The epoch-one ownership record was created conditionally by this
    /// activation. No earlier replica can exist for this cell.
    pub fresh: bool,
    /// This activation seized the cell from a DIFFERENT node. When false the
    /// ownership record still named us at `epoch - 1`, so no other process
    /// has written the cell since we evicted it and our preserved local
    /// state is authoritative.
    pub took_over: bool,
    /// Open the exact existing local epoch after a certified node-level
    /// handoff. This path performs no remote discovery or restore.
    pub resume_local: bool,
    /// The owner a takeover displaced, from the record version the acquire
    /// consumed. The node-log takeover interlock recovers that node's log
    /// before the restore reads or seals anything; `None` means the record
    /// was released or absent, which the release path already proved
    /// durable.
    pub prior: Option<String>,
}

pub struct ActivationResult {
    pub path: PathBuf,
    pub restored: bool,
    /// The paged VFS registered for this activation, when the restore paged
    /// instead of downloading. The local main-db file is then sparse, so every
    /// connection to `path` — the actor's included — must open through this
    /// VFS; a plain open reads holes as zeros and SQLite reports a malformed
    /// database (the 2026-09-01 cold-whale fleet failure).
    pub vfs: Option<String>,
}

/// Is this a preserved eviction snapshot? `.hibernated` is the pre-2026-08-05
/// name and is still recognised, or a node upgrading would keep those files
/// forever without counting them.
fn is_snapshot(path: &std::path::Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext == "evicted" || ext == "hibernated")
}

/// The preserved-snapshot cache, held in memory so the prune does not have to
/// walk the data directory to discover it has nothing to do.
///
/// `celld_logic::cache::plan_eviction` states the contract this exists to
/// honour: "An empty result means the cache already fits — the common case,
/// and it must never do I/O-worthy work then." The walk that used to feed it
/// broke that promise, because finding out the cache already fits meant
/// reading every directory under the watch root and stat-ing every entry in
/// them, once a minute, forever.
///
/// Keyed by path rather than by cell and epoch: a reactivation may consume the
/// *previous* epoch's snapshot, and either the current or the legacy name, so
/// the path is the only key that covers every case uniformly.
pub(crate) struct PreservedCache {
    filesystem: Arc<dyn celld_ltx::FileSystem>,
    entries: std::collections::BTreeMap<PathBuf, celld_logic::cache::CacheEntry>,
    bytes: u64,
    /// Prunes served since the index was last rebuilt from the filesystem.
    /// `None` until the first adoption or while a failed mutation requires a
    /// rescan.
    since_resync: Option<u32>,
    /// Empty epoch directories the last adoption walk found, with their
    /// modification time, for `sweep_empty_epoch_dirs`.
    empty_epoch_dirs: Vec<(PathBuf, i64)>,
}

/// The most preserved snapshots a node keeps, whatever their bytes.
///
/// The byte ceiling alone does not bound the files: a Durable Object that
/// stores one key leaves a 48 KiB snapshot, so the 2 GiB default held about
/// 44,000 of them, four inodes each with their directories. A 10 GiB cell has
/// 655,360 inodes. This keeps the cache to 16,384 snapshots (about 65,000
/// inodes); a cell whose snapshot was dropped restores from the bucket.
pub(crate) const MAX_PRESERVED_SNAPSHOTS: usize = 16_384;

/// An empty epoch directory the walk finds is removed only once it is this
/// old. An activation creates its directory before it restores into it, and
/// a remote restore can take a while; nothing takes an hour.
const EMPTY_EPOCH_DIR_AGE_MS: i64 = 60 * 60 * 1000;

/// Remove `directory`, then each parent while it is empty, stopping below
/// `watch`. A cell's files live in `<watch>/<cell>/ltx/e<epoch>/`, and nothing
/// removed those directories once their last file went: every Durable Object
/// a node ever held left three directories behind, and every reactivation one
/// more epoch directory, until the filesystem ran out of inodes and the node
/// could not write at all.
///
/// Only empty directories go (`rmdir`), so the first level that still holds
/// anything stops the walk: a directory an activation is filling already holds
/// its epoch directory or file. An activation that creates its directory while
/// this runs loses nothing either, because `create_dir_all` recreates a parent
/// that vanished under it. Returns how many directories went.
pub(crate) fn remove_empty_ancestors(
    filesystem: &dyn celld_ltx::FileSystem,
    watch: &std::path::Path,
    directory: &std::path::Path,
) -> usize {
    let mut removed = 0;
    let mut current = Some(directory);
    while let Some(dir) = current {
        if dir == watch || !dir.starts_with(watch) {
            break;
        }
        if filesystem.remove_empty_dir(dir).is_err() {
            break;
        }
        removed += 1;
        current = dir.parent();
    }
    removed
}

/// Is `path` a cell's epoch directory, `<watch>/<cell>/ltx/e<digits>`?
pub(crate) fn is_epoch_dir(watch: &std::path::Path, path: &std::path::Path) -> bool {
    let Ok(rest) = path.strip_prefix(watch) else {
        return false;
    };
    let parts: Vec<_> = rest.components().collect();
    parts.len() == 3
        && parts[1].as_os_str() == "ltx"
        && parts[2].as_os_str().to_str().is_some_and(|name| {
            name.len() > 1 && name.starts_with('e') && name[1..].bytes().all(|b| b.is_ascii_digit())
        })
}

/// Rebuild after this many prunes. The index is maintained at every point that
/// creates or consumes a snapshot, but a failed rename or a file removed from
/// outside would drift it. Over-counting is harmless -- the node evicts a
/// little early -- while under-counting would let the cache grow past its
/// ceiling unnoticed, which is the whole thing this bounds, so the truth is
/// re-read periodically. At the 60 s prune period this is hourly.
const RESYNC_EVERY: u32 = 60;

impl PreservedCache {
    pub(crate) fn new(filesystem: Arc<dyn celld_ltx::FileSystem>) -> Self {
        Self {
            filesystem,
            entries: std::collections::BTreeMap::new(),
            bytes: 0,
            since_resync: None,
            empty_epoch_dirs: Vec::new(),
        }
    }

    /// Record a snapshot this process just created.
    ///
    /// The size and mtime are read from the file rather than assumed: `rename`
    /// carries the original modification time, so stamping "now" here would
    /// quietly reorder the LRU against what the filesystem actually holds.
    pub(crate) fn insert(&mut self, path: PathBuf) -> std::io::Result<()> {
        let meta = match self.filesystem.metadata(&path) {
            Ok(meta) => meta,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // Reactivation can consume the renamed snapshot before this
                // bookkeeping call gets the cache lock. The absent path has no
                // bytes to account, and an older entry for the same path is now
                // stale.
                self.forget(&path);
                return Ok(());
            }
            Err(error) => {
                // The renamed snapshot exists, but this process cannot account
                // for its size. Force the next prune to rebuild the complete
                // index instead of treating the current byte total as truth.
                self.since_resync = None;
                return Err(error);
            }
        };
        let entry = celld_logic::cache::CacheEntry {
            last_used_ms: meta.modified_unix_millis,
            bytes: meta.len,
        };
        if let Some(previous) = self.entries.insert(path, entry) {
            self.bytes = self.bytes.saturating_sub(previous.bytes);
        }
        self.bytes = self.bytes.saturating_add(entry.bytes);
        Ok(())
    }

    /// Drop a snapshot that has been consumed or deleted.
    pub(crate) fn forget(&mut self, path: &std::path::Path) {
        if let Some(entry) = self.entries.remove(path) {
            self.bytes = self.bytes.saturating_sub(entry.bytes);
        }
    }

    /// Bring the cache under `max_bytes`, returning (kept, evicted, bytes).
    ///
    /// Answers without touching the filesystem whenever the cache already
    /// fits, which is the steady state.
    pub(crate) fn prune(
        &mut self,
        watch: &std::path::Path,
        max_bytes: u64,
    ) -> std::io::Result<(usize, usize, u64)> {
        let mut first_error = None;
        match self.since_resync {
            // First prune of the process: adopt whatever a previous process
            // left behind. The walk rebuilds from ground truth, so anything
            // inserted before now is corrected rather than double-counted.
            None | Some(RESYNC_EVERY..) => {
                if let Err(error) = self.adopt(watch) {
                    // Adoption is transactional, so its failure leaves only
                    // entries that this process inserted or a prior complete
                    // scan installed. Deleting any such cache entry is safe.
                    // Continue from that known lower bound, then report that
                    // the complete on-disk accounting remains unavailable.
                    first_error = Some(error);
                }
            }
            Some(count) => self.since_resync = Some(count + 1),
        }
        if self.bytes <= max_bytes && self.entries.len() <= MAX_PRESERVED_SNAPSHOTS {
            return match first_error {
                Some(error) => Err(error),
                None => Ok((self.entries.len(), 0, self.bytes)),
            };
        }
        let mut evicted = 0;
        // The count bound first: the oldest snapshots past it go, whatever
        // their size. A failed removal stays counted and is not retried here.
        if self.entries.len() > MAX_PRESERVED_SNAPSHOTS {
            let mut by_age: Vec<_> = self
                .entries
                .iter()
                .map(|(path, entry)| (entry.last_used_ms, path.clone()))
                .collect();
            by_age.sort();
            let excess = self.entries.len() - MAX_PRESERVED_SNAPSHOTS;
            for (_, path) in by_age.into_iter().take(excess) {
                match self.filesystem.remove_file(&path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                        continue;
                    }
                }
                self.forget(&path);
                evicted += 1;
                if let Some(parent) = path.parent() {
                    remove_empty_ancestors(&*self.filesystem, watch, parent);
                }
            }
        }
        let mut failed_paths = std::collections::BTreeSet::new();
        loop {
            // A failed deletion remains part of the byte total and is pinned
            // for this invocation. Replan the unfailed entries against the
            // remaining budget, so a failed planned victim makes another LRU
            // entry eligible instead of stopping progress above the ceiling.
            let mut pinned_bytes = 0_u64;
            let mut paths = Vec::new();
            let mut entries = Vec::new();
            for (path, entry) in &self.entries {
                if failed_paths.contains(path) {
                    pinned_bytes = pinned_bytes.saturating_add(entry.bytes);
                } else {
                    paths.push(path.clone());
                    entries.push(*entry);
                }
            }
            let remaining_budget = max_bytes.saturating_sub(pinned_bytes);
            let plan = celld_logic::cache::plan_eviction(&entries, remaining_budget);
            if plan.is_empty() {
                break;
            }
            for index in plan {
                match self.filesystem.remove_file(&paths[index]) {
                    Ok(()) => {
                        self.forget(&paths[index]);
                        evicted += 1;
                        if let Some(parent) = paths[index].parent() {
                            remove_empty_ancestors(&*self.filesystem, watch, parent);
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        self.forget(&paths[index]);
                        evicted += 1;
                    }
                    Err(error) => {
                        // Keep the failed entry and its bytes in the inventory.
                        // Excluding it from later plans guarantees termination,
                        // while replanning gives every remaining entry a chance
                        // to restore the known index to its byte ceiling.
                        failed_paths.insert(paths[index].clone());
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok((self.entries.len(), evicted, self.bytes)),
        }
    }

    /// Read one directory for an adoption without treating an aggregated
    /// `NotFound` as an empty directory.
    ///
    /// A queued descendant can disappear after its parent was read, which is a
    /// complete observation of that now-absent subtree. The direct backend can
    /// also aggregate one vanished entry into the directory-level result. If
    /// the directory still exists, retry once and then propagate the error so
    /// the caller retains its known index instead of under-counting the
    /// directory. A missing watch root is always an error because a first-start
    /// scan cannot infer a complete inventory from it.
    fn read_adoption_directory(
        &self,
        watch: &std::path::Path,
        directory: &std::path::Path,
    ) -> std::io::Result<Option<Vec<celld_ltx::HostDirEntry>>> {
        for attempt in 0..=1 {
            match self.filesystem.read_dir(directory) {
                Ok(entries) => return Ok(Some(entries)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    match self.filesystem.metadata(directory) {
                        Err(confirm) if confirm.kind() == std::io::ErrorKind::NotFound => {
                            if directory == watch {
                                return Err(error);
                            }
                            return Ok(None);
                        }
                        Ok(metadata) if metadata.is_dir && attempt == 0 => continue,
                        Ok(_) => return Err(error),
                        Err(confirm) => return Err(confirm),
                    }
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("the bounded directory retry returned from every branch")
    }

    /// Read the preserved snapshots off disk and replace the index with them.
    ///
    /// The only walk left, and it runs once per process plus a periodic
    /// resync. `file_type` answers the recursion question from the directory
    /// entry itself, so a `stat` is spent only on the files that are actually
    /// candidates -- the live databases, their WALs, their shared-memory
    /// indexes and the meta directories are skipped on their names.
    fn adopt(&mut self, watch: &std::path::Path) -> std::io::Result<()> {
        let mut entries = std::collections::BTreeMap::new();
        let mut bytes = 0_u64;
        let mut empty_epoch_dirs = Vec::new();
        let mut stack = vec![watch.to_path_buf()];
        while let Some(directory) = stack.pop() {
            let Some(read) = self.read_adoption_directory(watch, &directory)? else {
                continue;
            };
            if read.is_empty() && is_epoch_dir(watch, &directory) {
                if let Ok(meta) = self.filesystem.metadata(&directory) {
                    empty_epoch_dirs.push((directory.clone(), meta.modified_unix_millis));
                }
            }
            for item in read {
                let path = item.path;
                if item.is_dir {
                    stack.push(path);
                    continue;
                }
                if !is_snapshot(&path) {
                    continue;
                }
                let meta = match self.filesystem.metadata(&path) {
                    Ok(meta) => meta,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        // Reactivation or stale-live pruning removed this path
                        // after the directory read. It is absent from the final
                        // cache image, so skipping it preserves complete
                        // accounting.
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                let entry = celld_logic::cache::CacheEntry {
                    last_used_ms: meta.modified_unix_millis,
                    bytes: meta.len,
                };
                bytes = bytes.saturating_add(entry.bytes);
                entries.insert(path, entry);
            }
        }
        self.entries = entries;
        self.bytes = bytes;
        self.empty_epoch_dirs = empty_epoch_dirs;
        self.since_resync = Some(0);
        Ok(())
    }

    /// Remove the empty epoch directories the last walk found that are older
    /// than `EMPTY_EPOCH_DIR_AGE_MS`, with every parent they leave empty.
    /// Directories that a node before this fix left behind go here; the
    /// prune, the eviction and the activation keep new ones from piling up.
    /// Returns how many directories went.
    pub(crate) fn sweep_empty_epoch_dirs(
        &mut self,
        watch: &std::path::Path,
        now_unix_millis: i64,
    ) -> usize {
        let mut removed = 0;
        for (directory, modified) in std::mem::take(&mut self.empty_epoch_dirs) {
            if now_unix_millis.saturating_sub(modified) >= EMPTY_EPOCH_DIR_AGE_MS {
                removed += remove_empty_ancestors(&*self.filesystem, watch, &directory);
            }
        }
        removed
    }
}

/// Copy a live database into a standalone snapshot. SQLite's backup API
/// includes committed WAL state without checkpointing or interfering with the
/// replicator's ownership of the WAL.
pub(crate) fn sqlite_snapshot(
    source: &std::path::Path,
    destination: &std::path::Path,
    vfs: Option<&str>,
) -> anyhow::Result<()> {
    {
        let open = |path: &std::path::Path, flags, role| {
            #[cfg(celld_internal_tests)]
            {
                match vfs {
                    Some(vfs) => crate::fault::with_connection_role(role, || {
                        Connection::open_with_flags_and_vfs(path, flags, vfs)
                    }),
                    None => Connection::open_with_flags(path, flags),
                }
            }
            #[cfg(not(celld_internal_tests))]
            {
                let _ = role;
                debug_assert!(vfs.is_none());
                Connection::open_with_flags(path, flags)
            }
        };
        let source = open(
            source,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
            "sqlite_snapshot_source",
        )?;
        let mut destination = open(
            destination,
            rusqlite::OpenFlags::default(),
            "sqlite_snapshot_destination",
        )?;
        let backup = rusqlite::backup::Backup::new(&source, &mut destination)?;
        backup.run_to_completion(64, Duration::from_millis(5), None)?;
    }
    Ok(())
}

#[cfg(celld_internal_tests)]
#[doc(hidden)]
pub fn sqlite_snapshot_on_vfs_for_test(
    source: &std::path::Path,
    destination: &std::path::Path,
    vfs: &str,
) -> anyhow::Result<()> {
    sqlite_snapshot(source, destination, Some(vfs))
}

#[cfg(test)]
mod directory_tests {
    // Real directories through the direct filesystem: what a node's watch root
    // holds after evictions, prunes and reactivations.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use std::path::Path;

    fn fs() -> Arc<dyn celld_ltx::FileSystem> {
        Arc::new(celld_ltx::DirectFileSystem)
    }

    fn snapshot(watch: &Path, cell: &str, epoch: u64, bytes: usize) -> PathBuf {
        let dir = watch.join(cell).join("ltx").join(format!("e{epoch}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("db.evicted");
        std::fs::write(&path, vec![0_u8; bytes]).unwrap();
        path
    }

    fn count_dirs(root: &Path) -> usize {
        let mut n = 0;
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    n += 1;
                    stack.push(entry.path());
                }
            }
        }
        n
    }

    #[test]
    fn empty_ancestors_go_up_to_the_watch_root_and_stop_at_anything_left() {
        let tmp = tempfile::tempdir().unwrap();
        let watch = tmp.path().join("w");
        let lone = snapshot(&watch, "Obj:a", 1, 10);
        std::fs::remove_file(&lone).unwrap();
        assert_eq!(remove_empty_ancestors(&*fs(), &watch, lone.parent().unwrap()), 3);
        assert!(watch.is_dir(), "the watch root itself stays");
        assert!(!watch.join("Obj:a").exists());

        // A cell with a newer epoch keeps `ltx` and itself.
        let old = snapshot(&watch, "Obj:b", 1, 10);
        let new = snapshot(&watch, "Obj:b", 2, 10);
        std::fs::remove_file(&old).unwrap();
        assert_eq!(remove_empty_ancestors(&*fs(), &watch, old.parent().unwrap()), 1);
        assert!(new.is_file());
        // A directory that still holds a file is never removed.
        assert_eq!(remove_empty_ancestors(&*fs(), &watch, new.parent().unwrap()), 0);
        assert!(new.is_file());
        // Nothing outside the watch root, even if empty.
        let outside = tmp.path().join("elsewhere/e1");
        std::fs::create_dir_all(&outside).unwrap();
        assert_eq!(remove_empty_ancestors(&*fs(), &watch, &outside), 0);
        assert!(outside.is_dir());
    }

    #[test]
    fn only_cell_epoch_directories_are_epoch_dirs() {
        let w = Path::new("/w");
        assert!(is_epoch_dir(w, Path::new("/w/Obj:a/ltx/e1")));
        assert!(is_epoch_dir(w, Path::new("/w/Obj:a/ltx/e123")));
        for not in ["/w/Obj:a/ltx", "/w/Obj:a/ltx/e", "/w/Obj:a/ltx/e1x", "/w/Obj:a/other/e1", "/w/Obj:a/ltx/e1/meta", "/x/Obj:a/ltx/e1", "/w/e1"] {
            assert!(!is_epoch_dir(w, Path::new(not)), "{not}");
        }
    }

    #[test]
    fn a_pruned_snapshot_takes_its_directories_with_it() {
        let tmp = tempfile::tempdir().unwrap();
        let watch = tmp.path().join("w");
        let mut cache = PreservedCache::new(fs());
        for i in 0..20 {
            cache.insert(snapshot(&watch, &format!("Obj:{i:02}"), 1, 1000)).unwrap();
        }
        assert_eq!(count_dirs(&watch), 60);
        // 20 KB of snapshots under a 5 KB ceiling: 15 go, and their 45 directories.
        let (kept, evicted, _) = cache.prune(&watch, 5_000).unwrap();
        assert_eq!((kept, evicted), (5, 15));
        assert_eq!(count_dirs(&watch), 15);
    }

    #[test]
    fn the_snapshot_count_is_bounded_whatever_the_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let watch = tmp.path().join("w");
        let mut cache = PreservedCache::new(fs());
        cache.prune(&watch, u64::MAX).unwrap_err(); // no watch root yet: the first walk says so
        std::fs::create_dir_all(&watch).unwrap();
        cache.prune(&watch, u64::MAX).unwrap();
        let extra = 3;
        for i in 0..MAX_PRESERVED_SNAPSHOTS + extra {
            let path = watch.join(format!("Obj:{i:05}/ltx/e1/db.evicted"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
            cache.insert(path).unwrap();
        }
        let (kept, evicted, _) = cache.prune(&watch, u64::MAX).unwrap();
        assert_eq!((kept, evicted), (MAX_PRESERVED_SNAPSHOTS, extra));
        assert_eq!(count_dirs(&watch), 3 * MAX_PRESERVED_SNAPSHOTS);
    }

    #[test]
    fn the_walk_sweeps_old_empty_epoch_dirs_and_spares_new_ones() {
        let tmp = tempfile::tempdir().unwrap();
        let watch = tmp.path().join("w");
        // What a node before the fix left: pruned cells, a cell's old epochs,
        // and one activation that has just created its directory.
        for i in 0..5 {
            std::fs::create_dir_all(watch.join(format!("Obj:gone{i}/ltx/e1"))).unwrap();
        }
        for e in 1..4 {
            std::fs::create_dir_all(watch.join(format!("Obj:hot/ltx/e{e}"))).unwrap();
        }
        let live = snapshot(&watch, "Obj:hot", 4, 10);
        std::fs::create_dir_all(watch.join("node-log")).unwrap(); // not a cell: never touched
        let mut cache = PreservedCache::new(fs());
        cache.prune(&watch, u64::MAX).unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        // Just made: an activation may be about to restore into it.
        assert_eq!(cache.sweep_empty_epoch_dirs(&watch, now), 0);
        assert_eq!(count_dirs(&watch), 5 * 3 + 2 + 4 + 1); // gone cells; hot cell, ltx, e1-e4; node-log
        // An hour on, the next walk removes them.
        cache.since_resync = None;
        cache.prune(&watch, u64::MAX).unwrap();
        assert_eq!(cache.sweep_empty_epoch_dirs(&watch, now + EMPTY_EPOCH_DIR_AGE_MS), 5 * 3 + 3);
        assert!(live.is_file());
        assert!(watch.join("node-log").is_dir());
        assert_eq!(count_dirs(&watch), 3 + 1);
    }
}
