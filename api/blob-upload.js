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
  } catch (error) {
    return new Response(JSON.stringify({ error: 'Invalid upload payload.' }), {
      status: 400,
      headers: jsonHeaders(),
    });
  }

  try {
    const response = await handleUpload({
      body,
      request,
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
