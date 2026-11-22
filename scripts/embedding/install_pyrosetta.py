"""Install the official quarterly PyRosetta release in the active rule env."""
from pathlib import Path
import ssl
import subprocess
import sys
import urllib.request

VERSION = '2026.29+releasequarterly.80a0635615'
MIRROR = 'https://graylab.jhu.edu/download/PyRosetta4/archive/release-quarterly/release/'


def main():
    cache = Path('artifacts/cache/pyrosetta').resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # The official East mirror omits its intermediate certificate. Fetch its
    # public AIA certificate, then verify its signature against trusted roots
    # before adding it to a process-local bundle. Never disable TLS validation.
    intermediate = cache / 'intermediate.der'
    with urllib.request.urlopen('http://crt.sectigo.com/InCommonRSAOVSSLCA3.crt', timeout=30) as response:
        intermediate.write_bytes(response.read())
    pem = cache / 'intermediate.pem'
    subprocess.run(['openssl', 'x509', '-inform', 'DER', '-in', str(intermediate), '-out', str(pem)], check=True)
    roots = ssl.get_default_verify_paths().cafile
    if not roots:
        raise RuntimeError('No trusted CA bundle found in the rule environment')
    subprocess.run(['openssl', 'verify', '-CAfile', roots, str(pem)], check=True)
    bundle = cache / 'ca-bundle.pem'
    bundle.write_bytes(Path(roots).read_bytes() + b'\n' + pem.read_bytes())
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--cache-dir', str(cache / 'pip'),
                    '--cert', str(bundle), '--timeout', '120', '--retries', '3',
                    '--find-links', MIRROR, f'pyrosetta=={VERSION}'], check=True)


if __name__ == '__main__':
    main()
