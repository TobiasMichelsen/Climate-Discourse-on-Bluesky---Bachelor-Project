from atproto import models

_INTERESTED_RECORDS = {
    "app.bsky.feed.post": models.AppBskyFeedPost,  # Regular posts
    "app.bsky.feed.like": models.AppBskyFeedLike,  # Likes
    "app.bsky.feed.repost": models.AppBskyFeedRepost,  # Reposts
    "app.bsky.graph.follow": models.AppBskyGraphFollow,  # Follows
    "app.bsky.graph.block": models.AppBskyGraphBlock,  # Blocks
    "app.bsky.embed.images": models.AppBskyEmbedImages,  # Image attachments
    "app.bsky.embed.external": models.AppBskyEmbedExternal,  # External link previews
    "app.bsky.feed.repost.subject": models.AppBskyFeedRepost,  # Repost subject reference
    "app.bsky.feed.like.subject": models.AppBskyFeedLike,  # Like subject reference
    "app.bsky.graph.follow.subject": models.AppBskyGraphFollow,  # Follow subject reference
    "app.bsky.richtext.facet": models.AppBskyRichtextFacet,  # Rich text elements (mentions, hashtags)
}

def get_interested_records():
    """Returns the interested records dictionary."""
    print("🔍 DEBUG: INTERESTED_RECORDS =", _INTERESTED_RECORDS)  # Print to debug
    return _INTERESTED_RECORDS
