def fetch_cdx_data(URIR: str, from_date: str, to_date: str, outfile: str = None) -> str:
    """
    Fetches CDX data from the Wayback Machine API for the given URI-R and 
    writes it to a file. Returns the output filename.
    
    Parameters:
        urir (str): The URL to fetch CDX data for.
        from_date (str): The start date for the CDX data in YYYYMMDD format.
        to_date (str): The end date for the CDX data in YYYYMMDD format.
        outfile (str, optional): Output file path. If None, uses default naming.
    
    Returns:
        str: The name of the output file containing the CDX data.
    """
    import os
    from requests import Session
    from rich.console import Console
    from urllib.parse import urlencode
    
    #FROM = "20150424"
    #TO = "20151223"
    #OTHER_PARAMS = f"&from={from_date}&to={to_date}&collapse=timestamp:8&filter=statuscode:200"  # One entry per day, 200 OK
    OTHER_PARAMS = f"&from={from_date}&to={to_date}&filter=statuscode:200"  # All entries, 200 OK
    REQSESSION = Session()
    errprint = Console(stderr=True, style="red", highlight=False).print

    # Define output directory
    output_dir = "cdx_files"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Define output file path
    if outfile is None:
        outfile = os.path.join(output_dir, f"cdx-{from_date}-{to_date}.cdx")
    else:
        # Siempre guarda en cdx_files, aunque pases un nombre personalizado
        outfile = os.path.join(output_dir, os.path.basename(outfile))

    # Define a custom User-Agent header
    HEADERS = {
        "User-Agent": "ODU WS-DL research (jbote001@odu.edu)"
    }

    def get_stream_from_api(url):
        pages = int(REQSESSION.get(f"{url}&showNumPages=true", headers=HEADERS).text)
        for page in range(pages):
            pageurl = f"{url}&page={page}"
            errprint(f"Downloading [[cyan]{page + 1}/{pages}[/cyan]]: [magenta]{pageurl}[/magenta]")
            r = REQSESSION.get(pageurl, stream=True, headers=HEADERS)
            if r.ok:
                r.raw.decode_content = True
                for line in r.raw:
                    yield line

    def write_cdx(urir, cdxapi, params, outfile):
        url = f"{cdxapi}?{params}&{urlencode({'url': urir})}"
        input_stream = get_stream_from_api(url)
        
        timestamps = []
        
        with open(outfile, "w") as f:
            for line in input_stream:
                decoded_line = line.decode().strip()
                parts = decoded_line.split()
                if len(parts) > 1:
                    timestamp = parts[1]
                    timestamps.append(timestamp[:8])  # Store only YYYYMMDD
                    archive_url = f"https://web.archive.org/web/{timestamp}/{urir}"
                    f.write(decoded_line + " " + archive_url + "\n")
        
        try:
            input_stream.close()
        except:
            pass

        # Determine first and last available dates
        first_date = min(timestamps) if timestamps else None
        last_date = max(timestamps) if timestamps else None

        return first_date, last_date

    # API and parameters
    cdxapi = "https://web.archive.org/cdx/search"
    params = "matchType=exact" + OTHER_PARAMS

    # Fetch and write CDX data
    first_date, last_date = write_cdx(URIR, cdxapi, params, outfile)

    return outfile, first_date, last_date
