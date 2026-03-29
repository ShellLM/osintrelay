from . import bluesky, hackernews, mastodon, reddit, twitter, github

FETCHERS = {
    "twitter": twitter.fetch_data,
    "reddit": reddit.fetch_data,
    "bluesky": bluesky.fetch_data,
    "mastodon": mastodon.fetch_data,
    "hackernews": hackernews.fetch_data,
    "github": github.fetch_data,
}