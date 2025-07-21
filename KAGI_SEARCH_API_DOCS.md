# Kagi Search API Documentation for OpenSecret

## Overview

The OpenSecret platform provides a pass-through API to Kagi's search service with end-to-end encryption and JWT authentication. This document describes the API endpoint, request/response formats, and data structures.

## Authentication

All requests require:
1. A valid JWT token in the `Authorization` header: `Bearer <token>`
2. A session ID in the `x-session-id` header
3. Encrypted request body (for POST requests)

## Endpoint

```
POST /v1/search
```

## Request Format

### Headers
```
Authorization: Bearer <jwt_token>
x-session-id: <uuid>
Content-Type: application/json
```

### Request Body (Before Encryption)
```json
{
  "query": "your search query",
  "workflow": "search"  // optional, see workflows below
}
```

### Encrypted Request Format
The actual request body sent to the server must be encrypted:
```json
{
  "encrypted": "<base64_encoded_encrypted_data>"
}
```

### Available Workflows
- `"search"` - Standard web search (default)
- `"images"` - Image search
- `"videos"` - Video search
- `"news"` - News search
- `"podcasts"` - Podcast search

## Response Format

### Response Body (After Decryption)
```json
{
  "success": true,
  "data": {
    // Kagi API response data (see below)
  },
  "error": null
}
```

### Error Response
```json
{
  "success": false,
  "data": null,
  "error": "Error message"
}
```

### Encrypted Response Format
The actual response from the server will be encrypted:
```json
{
  "encrypted": "<base64_encoded_encrypted_response>"
}
```

## Kagi API Data Format

The `data` field in a successful response contains the Kagi search results:

### Top-Level Structure
```json
{
  "meta": {
    "trace": "trace-id-for-debugging",
    "id": "request-id",
    "node": "server-hostname",
    "ms": 123  // processing time in milliseconds
  },
  "data": {
    // Search results organized by type (see below)
  }
}
```

### Search Results Data Structure

The `data` object contains different arrays based on the result types. Each workflow may populate different fields:

```json
{
  "data": {
    "search": [...],           // Web search results
    "image": [...],            // Image results
    "video": [...],            // Video results
    "podcast": [...],          // Podcast results
    "podcast_creator": [...],  // Podcast creator results
    "news": [...],             // News article results
    "adjacent_question": [...],// Related questions
    "direct_answer": [...],    // Quick answers (calculations, conversions)
    "interesting_news": [...], // Kagi's curated news
    "interesting_finds": [...],// Small web discoveries
    "infobox": [...],          // Structured info about entities
    "code": [...],             // Code repositories and resources
    "package_tracking": [...], // Package tracking links
    "public_records": [...],   // Government/court documents
    "weather": [...],          // Weather information
    "related_search": [...],   // Related search suggestions
    "listicle": [...],         // List-based articles
    "web_archive": [...]       // Archived websites
  }
}
```

### Individual Search Result Structure

Each result in the arrays above follows this structure:

```json
{
  "url": "https://example.com/page",
  "title": "Page Title",
  "snippet": "A brief excerpt or description of the content...",
  "time": "2024-01-15T10:30:00Z",  // Optional: publication/update time
  "image": {                       // Optional: thumbnail/preview image
    "url": "https://example.com/image.jpg",
    "width": 200,
    "height": 150
  },
  "props": {                       // Optional: additional metadata
    // Varies by result type, examples:
    "author": "John Doe",
    "duration": "5:32",           // For videos/podcasts
    "question": "Related query?", // For adjacent_questions
    "price": "$19.99",            // For products
    // ... other type-specific properties
  }
}
```

## Example Usage

### Search Request Example

```javascript
// 1. Prepare the request data
const searchRequest = {
  query: "best programming languages 2024",
  workflow: "search"
};

// 2. Encrypt the request using your session key
const encryptedRequest = await encryptWithSessionKey(searchRequest, sessionId);

// 3. Send the request
const response = await fetch('https://api.opensecret.cloud/v1/search', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${jwtToken}`,
    'x-session-id': sessionId,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    encrypted: btoa(encryptedRequest)
  })
});

// 4. Decrypt the response
const encryptedResponse = await response.json();
const decryptedData = await decryptWithSessionKey(
  atob(encryptedResponse.encrypted),
  sessionId
);
```

### Image Search Example

```javascript
const imageSearchRequest = {
  query: "mountain landscape photography",
  workflow: "images"
};
```

### Response Example (Web Search)

```json
{
  "success": true,
  "data": {
    "meta": {
      "trace": "abc123",
      "id": "req_456",
      "node": "search-node-1",
      "ms": 89
    },
    "data": {
      "search": [
        {
          "url": "https://stackoverflow.com/questions/programming-languages-2024",
          "title": "What are the best programming languages to learn in 2024?",
          "snippet": "Based on industry trends and job market data, here are the top programming languages...",
          "time": "2024-01-10T15:20:00Z"
        },
        {
          "url": "https://github.com/trending",
          "title": "Trending Programming Languages on GitHub",
          "snippet": "See what the GitHub community is most excited about today...",
          "props": {
            "stars": "15.2k",
            "language": "Multiple"
          }
        }
      ],
      "related_search": [
        {
          "url": "#",
          "title": "programming languages for beginners 2024",
          "props": {
            "query": "programming languages for beginners 2024"
          }
        },
        {
          "url": "#",
          "title": "highest paying programming languages",
          "props": {
            "query": "highest paying programming languages"
          }
        }
      ]
    }
  },
  "error": null
}
```

## Error Handling

### Common Error Responses

1. **Missing API Key Configuration**
```json
{
  "success": false,
  "data": null,
  "error": "Search service not configured"
}
```

2. **Kagi API Error**
```json
{
  "success": false,
  "data": null,
  "error": "Search failed: <kagi_error_details>"
}
```

3. **Authentication Error** (HTTP 401)
```json
{
  "status": 401,
  "message": "Invalid JWT"
}
```

4. **Bad Request** (HTTP 400)
```json
{
  "status": 400,
  "message": "Bad Request"
}
```

## Rate Limiting

Rate limiting is enforced at multiple levels:
1. OpenSecret platform rate limits (based on your subscription)
2. Kagi API rate limits (passed through from Kagi)

## Notes for SDK Implementation

1. **Encryption/Decryption**: You must implement the session-based encryption/decryption using the same algorithm as other OpenSecret protected endpoints.

2. **Session Management**: The `x-session-id` must match an active session established during login.

3. **Error Handling**: Always check the `success` field first. If `false`, display the error message to the user.

4. **Result Type Handling**: Different workflows return results in different fields. For example:
   - `workflow: "search"` → results in `data.search`
   - `workflow: "images"` → results in `data.image`
   - `workflow: "news"` → results in `data.news`

5. **Optional Fields**: Many fields in the Kagi response are optional. Always check for existence before accessing.

6. **Props Field**: The `props` field contains type-specific metadata. Its structure varies based on the result type.

## Testing

To test the endpoint:

1. Authenticate and establish a session
2. Use the session key to encrypt a test search request
3. Send the encrypted request to `/v1/search`
4. Decrypt the response
5. Verify the structure matches the documentation

## Support

For issues related to:
- OpenSecret platform: Contact OpenSecret support
- Kagi search results/API: Refer to [Kagi API documentation](https://help.kagi.com/kagi/api/search.html)
- This integration: File an issue in the OpenSecret repository