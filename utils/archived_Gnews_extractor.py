def extract_data(archived_url, output_file="dataset/all_versions.json", max_retries=5):
    """
    Extracts news article JSON from a URL using GNews and appends it to a single JSON file with metadata.

    Args:
        archived_url (str): The URL of the archived page.
        output_file (str): Path to the single JSON file.

    Returns:
        str: Path to the saved file or an error message.
    """

    import os
    import random
    import hashlib
    import uuid
    from datetime import datetime
    from gnews import GNews
    import json
    import re
    import time

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Extracting date from Wayback Machine URL
    try:
        wayback_timestamp = archived_url.split("/")[4].split("id_")[0]
        warc_date = datetime.strptime(wayback_timestamp, "%Y%m%d%H%M%S").isoformat() + "Z"
        timestamp = wayback_timestamp  # Use this for filename
    except (IndexError, ValueError):
        return f"❌ Error: Could not extract valid timestamp from URL: {archived_url}"

    attempt = 0
    success = False
    article = None

    google_news = GNews()

    while attempt < max_retries:
        try:
            sleep_time = random.uniform(2, 10)
            print(f"⏳ Waiting {round(sleep_time, 5)} seconds before fetching {archived_url}...")
            time.sleep(sleep_time)

            # Use GNews to get the article object
            article = google_news.get_full_article(archived_url)
            if article and hasattr(article, "title"):
                success = True
                break
            else:
                print(f"⚠️ No article found or invalid response - Retrying {attempt+1}/{max_retries}")

        except Exception as e:
            print(f"⚠️ GNews fetch failed (attempt {attempt + 1}/{max_retries}): {e}")

        backoff_time = 2 ** attempt + random.uniform(1, 5)
        print(f"🔄 Retrying in {round(backoff_time, 2)} seconds...")
        time.sleep(backoff_time)
        attempt += 1

    if not success:
        return f"❌ Max retries ({max_retries}) exceeded for {archived_url}"

    # Build a dict from the article object
    data = {
        "title": getattr(article, "title", None),
        "text": getattr(article, "text", None),
        "authors": getattr(article, "authors", None)
        # "publish_date": article.publish_date.isoformat() if getattr(article, "publish_date", None) else None,
        # "images": list(getattr(article, "images", [])),
        # "url": getattr(article, "source_url", archived_url)
    }

    # Metadata
    content_length = len(data["text"].encode("utf-8")) if data["text"] else 0
    warc_record_id = f"<urn:uuid:{uuid.uuid4()}>"
    warc_block_digest = hashlib.sha1((data["text"] or "").encode("utf-8")).hexdigest()
    url_hash = hashlib.md5(archived_url.encode()).hexdigest()

    metadata = {
        "warc-date": warc_date,
        "warc-record-id": warc_record_id,
        "warc-block-digest": f"sha1:{warc_block_digest}",
        "warc-target-uri": archived_url,
        "content-length": content_length,
        "url-hash": url_hash
    }

    # Combine article data and metadata
    output = {
        "metadata": metadata,
        "article": data
    }

    # Append to a single JSON file (as a list of articles)
    try:
        # If file exists, load existing list, else start new
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                try:
                    articles_list = json.load(f)
                    if not isinstance(articles_list, list):
                        articles_list = []
                except Exception:
                    articles_list = []
        else:
            articles_list = []

        articles_list.append(output)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(articles_list, f, ensure_ascii=False, indent=4)
        print(f"✅ Article and metadata appended to {output_file}")
        return f"✅ Article JSON with metadata extracted and appended to {output_file}"
    except Exception as e:
        return f"❌ Error saving JSON for {archived_url}: {str(e)}"


