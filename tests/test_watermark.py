"""The shared watermark contract: SECDED codec, issuer partition, allocation.

Pure-Python throughout — the AudioSeal embed/detect machinery is exercised
in the App and Platform suites, where the heavy dependencies live.
"""

import pytest

from khaos_attribution.watermark import (
    PAYLOAD_RANGES,
    PAYLOAD_SPACE,
    allocate_watermark_id,
    decode_codeword,
    encode_payload,
    watermark_issuer,
)


def test_partition_covers_space_without_overlap():
    covered = sorted(p for r in PAYLOAD_RANGES.values() for p in r)
    assert covered == list(range(PAYLOAD_SPACE))


def test_codec_roundtrip_every_payload():
    seen = set()
    for payload in range(PAYLOAD_SPACE):
        codeword = encode_payload(payload)
        assert 0 <= codeword <= 0xFFFF  # provenance schema range
        assert codeword not in seen
        seen.add(codeword)
        assert decode_codeword(codeword) == (payload, codeword, False)


def test_codec_corrects_any_single_bit_error():
    for payload in range(0, PAYLOAD_SPACE, 89):
        codeword = encode_payload(payload)
        for bit in range(16):
            decoded = decode_codeword(codeword ^ (1 << bit))
            assert decoded is not None
            assert decoded[0] == payload
            assert decoded[2] is True


def test_codec_refuses_double_bit_errors():
    for payload in (0, 1, 682, 1365, PAYLOAD_SPACE - 1):
        codeword = encode_payload(payload)
        for i in range(16):
            for j in range(i + 1, 16):
                assert decode_codeword(codeword ^ (1 << i) ^ (1 << j)) is None


def test_issuer_recovered_from_codeword():
    assert watermark_issuer(encode_payload(0)) == "platform"
    assert watermark_issuer(encode_payload(1023)) == "platform"
    assert watermark_issuer(encode_payload(1024)) == "app"
    assert watermark_issuer(encode_payload(2047)) == "app"
    # a corrupted word is not an issuer claim
    assert watermark_issuer(encode_payload(7) ^ 0b1) is None


def test_allocation_respects_partition_and_exhaustion():
    for issuer, payload_range in PAYLOAD_RANGES.items():
        allocated = allocate_watermark_id(set(), issuer=issuer)
        decoded = decode_codeword(allocated)
        assert decoded is not None and decoded[0] in payload_range

        partition = {encode_payload(p) for p in payload_range}
        victim = encode_payload(payload_range.start + 7)
        assert allocate_watermark_id(partition - {victim}, issuer=issuer) == victim
        assert allocate_watermark_id(partition, issuer=issuer) is None

    # exhausting one partition never bleeds into the other
    platform_full = {encode_payload(p) for p in PAYLOAD_RANGES["platform"]}
    app_id = allocate_watermark_id(platform_full, issuer="app")
    assert decode_codeword(app_id)[0] in PAYLOAD_RANGES["app"]


def test_unknown_issuer_is_an_error():
    with pytest.raises(KeyError):
        allocate_watermark_id(set(), issuer="mystery")


# ---- watermarking v2: model-level allocation (docs/watermarking-v2.md) ----

from khaos_attribution.watermark import (  # noqa: E402
    MACHINE_BANDS,
    allocate_model_watermark_id,
    legacy_watermark_ids,
)


def test_bands_tile_the_payload_space_exactly():
    seen = set()
    for band_range in MACHINE_BANDS.values():
        assert not (seen & set(band_range)), "bands overlap"
        seen |= set(band_range)
    assert seen == set(range(PAYLOAD_SPACE)), "bands must tile 0..2047"


def test_model_allocation_stays_in_band_and_avoids_legacy():
    legacy_payloads = {decode_codeword(c)[0] for c in legacy_watermark_ids()}
    for band, band_range in MACHINE_BANDS.items():
        cid = allocate_model_watermark_id(band, set())
        payload, _, _ = decode_codeword(cid)
        assert payload in band_range
        assert payload not in legacy_payloads

    # dense band: the only free non-legacy payload is found deterministically
    band_range = MACHINE_BANDS["laptop"]
    free = next(p for p in band_range if p not in legacy_payloads)
    used = {encode_payload(p) for p in band_range if p != free}
    assert allocate_model_watermark_id("laptop", used) == encode_payload(free)
    assert allocate_model_watermark_id("laptop",
                                       used | {encode_payload(free)}) is None


def test_legacy_ids_are_frozen_valid_codewords():
    legacy = legacy_watermark_ids()
    assert len(legacy) >= 23
    for codeword in legacy:
        decoded = decode_codeword(codeword)
        assert decoded is not None and decoded[1] == codeword, \
            f"legacy {codeword} is not a clean codeword"


def test_unknown_band_is_an_error():
    with pytest.raises(ValueError):
        allocate_model_watermark_id("mystery", set())
