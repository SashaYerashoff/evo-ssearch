from embedding_space import (
    embedding_space_fingerprint,
    embedding_spaces_match,
    identified_embedding_space,
)


def test_fingerprint_changes_for_model_revision_and_dimension():
    base = {
        "backend": "siglip2",
        "model": "google/siglip2-base-patch16-224",
        "revision": "a",
        "dimension": 768,
    }
    assert embedding_space_fingerprint(base) != embedding_space_fingerprint(
        {**base, "revision": "b"}
    )
    assert embedding_space_fingerprint(base) != embedding_space_fingerprint(
        {**base, "dimension": 512}
    )


def test_siglip_requires_explicit_matching_identity():
    expected = identified_embedding_space(
        {
            "backend": "siglip2",
            "model": "google/siglip2-base-patch16-224",
            "revision": "rev-1",
            "dimension": 768,
        }
    )
    assert not embedding_spaces_match(expected, {})
    assert not embedding_spaces_match(
        expected,
        {**expected, "revision": "rev-2"},
    )
    assert embedding_spaces_match(expected, expected)


def test_legacy_openai_clip_without_metadata_remains_readable_only_by_clip():
    assert embedding_spaces_match(
        {"backend": "openai_clip", "model": "ViT-B/32", "dimension": 512},
        {},
    )
    assert not embedding_spaces_match(
        {"backend": "siglip2", "model": "siglip2", "dimension": 512},
        {},
    )
