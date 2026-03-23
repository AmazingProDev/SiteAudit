import { handleUpload } from '@vercel/blob/client';

export async function POST(request) {
  let body;

  try {
    body = await request.json();
  } catch (error) {
    return Response.json({ error: 'Invalid upload payload.' }, { status: 400 });
  }

  try {
    const jsonResponse = await handleUpload({
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

    return Response.json(jsonResponse);
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Blob upload failed.' },
      { status: 400 },
    );
  }
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
    },
  });
}
