import { handleUpload } from '@vercel/blob/client';

function jsonHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Content-Type': 'application/json; charset=utf-8',
  };
}

export default async function handler(request) {
  console.log('blob-upload request start', {
    method: request.method,
    url: request.url,
  });

  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: jsonHeaders(),
    });
  }

  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed.' }), {
      status: 405,
      headers: jsonHeaders(),
    });
  }

  let body;
  try {
    body = await request.json();
    console.log('blob-upload request parsed', {
      keys: body && typeof body === 'object' ? Object.keys(body) : [],
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Invalid upload payload.' }), {
      status: 400,
      headers: jsonHeaders(),
    });
  }

  try {
    console.log('blob-upload handleUpload start');
    const response = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname) => {
        console.log('blob-upload onBeforeGenerateToken', { pathname });
        const safePathname = String(pathname || 'upload.xlsx').replace(/[^a-zA-Z0-9._/-]+/g, '-');
        return {
          allowedContentTypes: [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/octet-stream',
          ],
          addRandomSuffix: true,
          callbackUrl: new URL('/api/blob-upload', request.url).toString(),
          tokenPayload: JSON.stringify({ kind: 'ssv-workbook' }),
          pathname: `ssv-uploads/${safePathname}`,
        };
      },
      onUploadCompleted: async ({ blob, tokenPayload }) => {
        console.log('blob-upload onUploadCompleted', {
          url: blob?.url,
          tokenPayload,
        });
      },
    });
    console.log('blob-upload handleUpload resolved');

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: jsonHeaders(),
    });
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : 'Blob upload failed.',
      }),
      {
        status: 500,
        headers: jsonHeaders(),
      },
    );
  }
}
