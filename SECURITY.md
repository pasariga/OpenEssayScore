# Security and privacy

Never post credentials or student records in public issues. Report a source vulnerability through GitHub's available private security-reporting facility; if unavailable, first open a minimal issue requesting a private channel, without disclosing exploit details or personal data.

Joblib/pickle can execute code. Only use operator-trusted, hash-verified artifacts; hashes detect changes but do not establish trust. The shipped loader pins historical v1 hashes. Optional research review bundles additionally require a trusted private provenance; their self-contained manifests are integrity checks, not signatures. Keep serving on loopback or a controlled authenticated network. No public auth/TLS or hostile-input isolation certification is claimed.

The backup scanner is fail-closed for known secret formats, private paths, nonempty CSV and unapproved binaries. Manual review remains necessary for prose, arbitrary secrets and model memorization. Private GitHub is not a substitute for data authorization. Revoke accidentally exposed credentials through their owner; do not paste them into reports. Do not delete original local/cloud backups after a successful upload.
