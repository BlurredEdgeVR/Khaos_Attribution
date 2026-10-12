# The key ceremony

The register countersigns every record with one Ed25519 key. Until the ceremony it runs on a development key and says so on every page. The ceremony makes the Guardians' key, puts it on the server, and records in the ledger who was there.

What the code can and cannot do, said plainly: the service signs from a PEM file it reads at start. It cannot sign from a hardware token. So the key is made on a machine that is offline for the whole ceremony and is written to removable media, never to that machine's own disk. The plan's "hardware key" is, for now, a sealed USB stick.

## The short ceremony

Two of the Foundation's directors, one new USB stick, one envelope, about fifteen minutes. The key is made on the server, where it lives anyway; the stick is its only other copy. Every step runs on the server as root, except the copy, which runs on a laptop with the stick in it.

1. Make the key. The register keeps signing with the development key meanwhile.
   ```
   register-key make
   ```
   It prints the key id, the public key and the private file's sha256. Write the id and the hash on a sheet.
2. Copy the private key straight onto the stick, and check it:
   ```
   ssh register 'sudo cat /srv/register/data/staged/register-key.pem' > /Volumes/<STICK>/register-key.pem
   shasum -a 256 /Volumes/<STICK>/register-key.pem
   ```
   The hash must match the sheet. Eject the stick, seal it in the envelope, both sign across the flap and the sheet.
3. Switch, naming who was there:
   ```
   register-key switch "Name, Name"
   ```
   It retires the development key, signs with the new one, checks the server now says so, and writes the names and the key id into the public ledger. If any check fails it puts the development key back.

Each step refuses to run twice, and neither runs once the development setting is gone, so the development key can never be promoted by deleting the setting alone. When more Guardians are named later, the same key is kept and their custody is recorded with `register-cli key --present "Name, Name" --note "custody"`.

The fuller ceremony below keeps the key off the server's own tools while it is made. Use it when the Guardians want that.

## The full ceremony: before

- The Guardians named and present. At least two.
- A laptop that can be started from a live Linux USB with networking off, with Python 3.12 available on it, and the contract's `[registry]` extra installed on that live system beforehand (the ceremony itself is offline).
- Three new USB sticks, labelled SERVER, BACKUP, PUBLIC.
- The register server reachable from another machine, with `REGISTER_DEV_KEY=1` still set.

## The ceremony

1. Start the offline laptop. Confirm networking is off. Mount the three sticks.
2. Make the key straight onto the SERVER stick:
   ```
   python -m register_service make-key /media/SERVER/register-key.pem
   ```
   It prints the key id and the public key. The private half is at `register-key.pem`, the public half beside it as `register-key.pub`. Nothing was written to the laptop.
3. Copy `register-key.pem` and `register-key.pub` to the BACKUP stick. Copy `register-key.pub` alone to the PUBLIC stick. Compare the three copies by hash:
   ```
   sha256sum /media/*/register-key.*
   ```
4. Write the key id and the public key's sha256 on the ceremony sheet. Every Guardian signs the sheet.
5. Seal the BACKUP stick in an envelope signed across the flap by every Guardian. It is opened only to restore the server's key, by two Guardians together.
6. Shut the laptop down. The live system leaves nothing behind.

## On the server, the same day

1. Place the key, as root:
   ```
   install -o register -g register -m 600 /media/SERVER/register-key.pem /srv/register/data/register-key.pem
   install -o register -g register -m 644 /media/SERVER/register-key.pub /srv/register/data/register-key.pub
   sed -i '/^REGISTER_DEV_KEY=/d' /etc/register.env
   systemctl restart register.service
   ```
2. Check the page no longer says development, and the key id matches the sheet:
   ```
   curl -fsS https://register.khaos.dev/api/public-key
   ```
3. Record the ceremony as the ledger's entry:
   ```
   register-cli key --present "Name, Name" --note "first ceremony"
   ```
   The entry carries the key id, the public key's hash and the names. It is in the public ledger from then on.
4. Wipe the SERVER stick.

## After

- Every Space pins the public key from the PUBLIC stick (or from `/api/public-key`, compared against the sheet) in its `[register] public_key_pem`. An exposed Space serves nothing countersigned by the development key from then on, so anything published before the ceremony is published again.
- Every Workshop that enrolled on the development key runs `scripts/register.py pin` to pin the new key.

## What would change this

A signing step that can use a hardware token (PKCS#11 or an agent) would let the private key never exist as a file at all. It is not written. Until it is, the BACKUP envelope is the only copy of the key that is not on the server.
