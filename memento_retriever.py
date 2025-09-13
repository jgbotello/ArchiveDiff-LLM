from utils import CDX_fetcher
from utils import archived_Gnews_extractor
import os
import time
import hashlib
import random

# List of URLs to analyze
urls = [
    "https://www.nytimes.com/2012/06/18/world/europe/greek-elections.html",
    "https://www.nytimes.com/2012/06/19/world/syria-dominates-as-obama-and-putin-meet.html",
    "https://www.nytimes.com/2012/06/20/world/asia/insurgents-strike-checkpoint-in-southern-afghanistan.html",
    "https://www.nytimes.com/2012/06/19/us/politics/road-trip-helps-romney-brush-up-on-banter.html",
    "https://www.nytimes.com/2012/06/17/world/asia/activism-grows-as-singapore-loosens-restrictions.html",
    "https://www.nytimes.com/2012/09/13/us/politics/behind-romneys-decision-to-criticize-obama-on-libya.html",
    "https://www.nytimes.com/2012/06/18/world/middleeast/egyptian-presidential-vote-enters-second-day.html",
    "https://www.nytimes.com/2012/06/17/world/asia/in-shift-china-stifles-debate-on-economic-change.html",
    "https://www.nytimes.com/2012/10/21/world/iran-said-ready-to-talk-to-us-about-nuclear-program.html",
    "https://www.nytimes.com/2012/06/20/world/middleeast/mubarak-is-on-life-support-egypt-security-officials-say.html",
    "http://www.nytimes.com/2013/06/07/opinion/president-obamas-dragnet.html",
    "https://www.nytimes.com/2012/06/18/us/rodney-king-whose-beating-led-to-la-riots-dead-at-47.html",
    "https://www.nytimes.com/2012/12/15/nyregion/shooting-reported-at-connecticut-elementary-school.html",
    "https://www.nytimes.com/2013/03/31/science/space/yvonne-brill-rocket-scientist-dies-at-88.html",

]

from_date = "20110101"
to_date = "20151230"

MAX_CAPTURES = int(os.getenv("MAX_CAPTURES", "20000"))

start_time = time.time()

def get_dataset_title(url):
    slug = url.rstrip('/').split('/')[-1]
    slug = slug.split('.')[0]
    words = slug.split('-')
    title = '-'.join(words[:3]) if len(words) >= 3 else slug
    return title

for urir in urls:
    dataset_title = get_dataset_title(urir)
    cdx_outfile = f"{dataset_title}_cdx.txt"
    output_file = os.path.join("dataset", f"{dataset_title}_all_versions.json")

    # Get the CDX data
    outfile, first_date, last_date = CDX_fetcher.fetch_cdx_data(urir, from_date, to_date, outfile=cdx_outfile)

    print(f"CDX data for {dataset_title} written to: {outfile}")
    if first_date and last_date:
        print(f"First available capture: {first_date}")
        print(f"Last available capture: {last_date}")
    else:
        print(f"No captures found for {dataset_title} in the given date range.")
        continue

    if not os.path.exists(outfile) or os.path.getsize(outfile) == 0:
        print(f"❌ Error: CDX file for {dataset_title} is empty or missing.")
        continue

    all_wayback_links = []
    with open(outfile, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                all_wayback_links.append(parts[-1])

    total_found = len(all_wayback_links)
    if total_found == 0:
        print(f"⚠️  No Wayback URLs found inside CDX for {dataset_title}.")
        continue

    wayback_links = all_wayback_links[:MAX_CAPTURES]
    print(f"✅ Found {total_found} Wayback URLs for {dataset_title}. Processing {len(wayback_links)} (limit={MAX_CAPTURES}).")

    # Extract Text from Mementos
    for url in wayback_links:
        result = archived_Gnews_extractor.extract_data(url, output_file)
        print(result)

    print(f"✅ Selected archived pages for {dataset_title} have been extracted and saved in '{output_file}'!")

    delay = random.uniform(0.5, 5.0)
    print(f"⏳ Waiting {delay:.2f} seconds before the next request...")
    time.sleep(delay)

end_time = time.time()
execution_time = end_time - start_time
print(f"🚀 The crawling process took {execution_time:.4f} seconds.")