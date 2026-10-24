# On-demand backup, versioning and recovery

No recurring task is installed. `tools/backup.py` uses an existing authenticated GitHub CLI and HTTPS Git. It never reads a token or changes global Git credential configuration. Authenticate interactively yourself if needed. A private repository does not authorize copying student data.

> Visibility check (2026-10-24): `pasariga/OpenEssayScore-Archive` is currently public and has existing published Release assets. The private-archive workflow below describes the original intended storage model, not the current GitHub access controls. Confirm the intended policy separately; this documentation change does not alter visibility or publish any artifact.

## Reviewed sources and profile

Maintain three separate directories: public source checkout, private archive metadata checkout, and a manually selected/sanitized private payload. Do not point payload at an entire research workspace or old mixed archive. Payload includes permitted model files, code, configurations, aggregate reports and export provenance; no raw records, individual embeddings/predictions, upstream backbone or credentials. Hash approval of a binary records a human-reviewed selection, not an automatic privacy proof.

Create an untracked `local-config.json` outside the repositories:

```json
{
  "public_dir":"path/to/OpenEssayScore",
  "archive_dir":"path/to/OpenEssayScore-Archive",
  "payload_dir":"path/to/reviewed-payload",
  "output_dir":"path/to/backup-output",
  "approved_binaries":"path/to/reviewed-binary-hashes.json",
  "public_repo":"OWNER/OpenEssayScore",
  "archive_repo":"OWNER/OpenEssayScore-Archive"
}
```

The binary hash JSON maps each relative payload path to its approved SHA-256. Never fill it by blindly approving every new file; review additions/transforms first. The utility checks repository visibility, source and payload content, refuses unapproved binaries/nonempty CSV/nested archives, and creates an immutable timestamped ZIP containing a complete SHA-256 manifest. It commits code/metadata, pushes without force, uploads private Release assets, re-downloads them and compares archive and internal hashes. Models are kept out of regular Git history. GitHub [warns above50MiB and blocks above100MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github); use separate Release assets and stay within current release limits. No paid LFS storage is configured.

```bash
python tools/backup.py sync --profile /your/local-config.json --tag backup-YYYYMMDDTHHMMSSZ
```

New changes require a new timestamp tag. Retry the same tag only for the same frozen payload; a different existing asset/hash is rejected, never overwritten. Network pushes/uploads get up to3 retries. Diverged Git history fails for review; no force/reset, history deletion or pruning. A rerun verifies existing assets and skips identical ones. Source-only changes can create a normal Git commit separately; backup tags never promote models to stable releases.

## Restore and verify

```bash
gh repo clone OWNER/OpenEssayScore clean-source
gh repo clone OWNER/OpenEssayScore-Archive clean-metadata
gh release download BACKUP_TAG --repo OWNER/OpenEssayScore-Archive --dir downloaded
python clean-source/tools/backup.py verify downloaded/BACKUP_TAG-research-archive.zip --restore clean-payload
# In a compatible authorized CPU environment, install clean-source with .[http,test].
python clean-source/tools/restore_smoke.py --payload clean-payload --output restore-result.json
```

Before extraction compare the ZIP's SHA-256 with the separately downloaded checksum and versioned private metadata. `verify` checks exact members, paths, sizes and every hash and requires an empty destination. The utility's own manifest is an integrity mechanism, not a signature or permission grant. Then test Python/CLI/HTTP parity using technical controls, not fabricated human ratings. Full original-data/R4 reconstruction requires separately retained authorized research storage; the sanitized archive intentionally cannot supply excluded records.

Retain all existing local and DSW originals, each backup tag, export transformation records, model/file hashes and recovery receipts. GitHub is one additional copy, not the sole disaster-recovery strategy. Periodically verify another controlled offline copy on demand. Do not enable automatic stable releases or long-running background tasks.
