# Data Dictionary

## Video Data Files (CSVs)

| Column | Type | Description |
|--------|------|-------------|
| video_id | String | Unique identifier for the video |
| trending_date | String | Date when video was trending (YY.DD.MM format) |
| title | String | Video title |
| channel_title | String | Name of the YouTube channel |
| category_id | String | Category ID (maps to category JSON) |
| publish_time | String | Video publication timestamp |
| tags | String | Video tags (pipe-separated) |
| views | String | Number of views |
| likes | String | Number of likes |
| dislikes | String | Number of dislikes |
| comment_count | String | Number of comments |
| thumbnail_link | String | URL to video thumbnail |
| comments_disabled | String | Whether comments are disabled |
| ratings_disabled | String | Whether ratings are disabled |
| video_error_or_removed | String | Whether video has errors or was removed |
| description | String | Video description |

## Category Data Files (JSONs)

Contains mapping of category IDs to category names for each country.

Structure:
```json
{
  "items": [
    {
      "id": "1",
      "snippet": {
        "title": "Film & Animation"
      }
    }
  ]
}
```

## Countries Available

- CA: Canada
- DE: Germany
- FR: France
- GB: Great Britain
- IN: India
- JP: Japan
- KR: South Korea
- MX: Mexico
- RU: Russia
- US: United States
