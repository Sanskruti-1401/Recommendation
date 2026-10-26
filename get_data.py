"""Download public-domain Project Gutenberg novels (via the NLTK data mirror) into data/gutenberg/."""
import io, os, urllib.request, zipfile
URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"
os.makedirs("data", exist_ok=True)
print("downloading", URL)
zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())).extractall("data")
print("done ->", sorted(os.listdir("data/gutenberg")))
