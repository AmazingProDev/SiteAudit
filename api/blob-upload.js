const { handleUpload } = require('@vercel/blob/client');

function sendJson(res, status, payload) {
  res.statusCode = status;
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(payload));
}

module.exports = async function handler(req, res) {
  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.end();
    return;
  }

  if (req.method !== 'POST') {
    sendJson(res, 405, { error: 'Method not allowed.' });
    return;
  }

  let body = req.body;

  if (typeof body === 'string') {
    try {
      body = JSON.parse(body);
    } catch (error) {
      sendJson(res, 400, { error: 'Invalid upload payload.' });
      return;
    }
  }

  if (!body || typeof body !== 'object') {
    sendJson(res, 400, { error: 'Invalid upload payload.' });
    return;
  }

  try {
    const jsonResponse = await handleUpload({
      body,
      request: req,
      onBeforeGenerateToken: async (pathname) => {
        const safePathname = String(pathname || 'upload.xlsx').replace(/[^a-zA-Z0-9._/-]+/g, '-');
        return {
          allowedContentTypes: [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/octet-stream',
          ],
          addRandomSuffix: true,
          tokenPayload: JSON.stringify({ kind: 'ssv-workbook' }),
          pathname: `ssv-uploads/${safePathname}`,
        };
      },
      onUploadCompleted: async () => {},
    });

    sendJson(res, 200, jsonResponse);
  } catch (error) {
    const message =
      error && typeof error.message === 'string'
        ? error.message
        : 'Blob upload failed.';
    sendJson(res, 500, { error: message });
  }
};
