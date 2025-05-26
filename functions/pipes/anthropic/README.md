# Anthropic Manifold Pipe

A simple Python interface to interact with Anthropic language models via the Open Web UI manifold pipe system.

---

## Features

- Supports multiple Anthropic models.
- Handles text and image inputs (with size validation).
- Supports streaming and non-streaming responses.
- Easy to configure with your Anthropic API key.

---

## Notes

- Images must be under 5MB each and total under 100MB for a request.
- The model name should be prefixed with `"anthropic."` (e.g., `"anthropic.claude-3-opus-20240229"`).
- Errors during requests are returned as strings.

---

## License

MIT License

---

## Authors

- justinh-rahb ([GitHub](https://github.com/justinh-rahb))
- christian-taillon