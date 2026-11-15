from bam_rag.corpus import parse_article_pages


def test_extracts_articles_and_tracks_source_pages() -> None:
    pages = [
        (19, "Table of contents\nArticle 1"),
        (
            20,
            "1. STATUT DE BANK AL-MAGHRIB\nPUBLIC\n"
            "Dahir n°1-19-82 portant statut de Bank Al-Maghrib\n"
            "Article premier\nLa Banque exerce ses missions.\n"
            "Article 2\nLe capital est fixé à 500 dirhams.",
        ),
        (21, "PUBLIC\n21\nArticle 3\nLes actes sont régis par la loi."),
    ]

    articles = parse_article_pages(pages)

    assert len(articles) == 3
    assert articles[0].article_number == "1er"
    assert articles[0].document_title.startswith("Dahir n°1-19-82")
    assert articles[0].page_start == 20
    assert articles[1].page_start == 20
    assert articles[2].page_start == 21
    assert articles[2].page_end == 21
    assert "Les actes sont régis" in articles[2].text


def test_repeated_article_numbers_get_distinct_ids() -> None:
    pages = [
        (20, "Loi n° 10-20 relative aux banques\nArticle 1\nPremière règle."),
        (21, "Circulaire n° 3/2024 sur les banques\nArticle 1\nDeuxième règle."),
    ]

    articles = parse_article_pages(pages)

    assert len(articles) == 2
    assert articles[0].article_id != articles[1].article_id


def test_handles_superscript_footnotes_and_inline_article_titles() -> None:
    pages = [
        (
            20,
            "Loi n° 103-12\nArticle 3154\nTexte de l’article trente et un.\n"
            "Article 19 bis 5\nTexte de l’article dix-neuf bis.\n"
            "Article premier : Participation au système\nLe texte suit.",
        )
    ]

    articles = parse_article_pages(pages)

    assert [article.article_number for article in articles] == ["31", "19 bis", "1er"]
    assert "Participation au système" in articles[2].text


def test_handles_subarticles_suffixes_and_page_number_collisions() -> None:
    pages = [
        (
            199,
            "Loi n° 1-99\nArticle 218-4-1 188\nUne disposition.\n"
            "Article 9.1199\nUne autre disposition.\n"
            "Article 6 sexies362\nDernière disposition.",
        )
    ]

    articles = parse_article_pages(pages)

    assert [article.article_number for article in articles] == [
        "218-4-1",
        "9.1",
        "6 sexies",
    ]


def test_preserves_article_numbers_ending_with_the_page_number() -> None:
    articles = parse_article_pages(
        [(90, "Article 190\nPremière règle.\nArticle 7-190\nDeuxième règle.")]
    )

    assert [article.article_number for article in articles] == ["190", "7-190"]


def test_long_document_titles_do_not_collide_in_article_ids() -> None:
    prefix = "Arrêté conjoint du ministre de l’industrie du commerce et de l’économie "
    pages = [
        (20, f"{prefix}n° 1\nArticle 1\nPremière règle."),
        (21, f"{prefix}n° 2\nArticle 1\nDeuxième règle."),
    ]

    articles = parse_article_pages(pages)

    assert articles[0].article_id != articles[1].article_id


def test_lowercase_legal_reference_does_not_replace_document_title() -> None:
    pages = [
        (20, "Loi n° 1-20 portant statut\nArticle 1\nPremière règle."),
        (21, "loi n° 99-99 citée dans le texte\nArticle 2\nDeuxième règle."),
    ]

    articles = parse_article_pages(pages)

    assert articles[0].document_title == articles[1].document_title


def test_section_heading_is_attached_to_the_following_article() -> None:
    pages = [
        (
            20,
            "Loi n° 1-20\nArticle 1\nPremière règle.\n"
            "Chapitre II\nMissions\nArticle 2\nDeuxième règle.",
        )
    ]

    articles = parse_article_pages(pages)

    assert "Missions" not in articles[0].text
    assert articles[1].section_context == "Chapitre II — Missions"
