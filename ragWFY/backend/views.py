from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from .rag_web import query_website
from .rag_youtube import process_youtube, ask_question
from .rag_file import query_file
import tempfile
import os


@csrf_exempt
@require_POST
def api_web_chat(request):
    try:
        website_url = request.POST.get('url')
        question = request.POST.get('question')
        if not website_url or not question:
            return JsonResponse({'error': 'url and question are required'}, status=400)
        answer = query_website(website_url, question)
        return JsonResponse({'answer': answer})
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)


@csrf_exempt
@require_POST
def api_youtube_chat(request):
    try:
        url = request.POST.get('url')
        question = request.POST.get('question', '')
        if not url:
            return JsonResponse({'error': 'url is required'}, status=400)
        result = process_youtube(url)
        transcript = result.get('transcript', '')
        if question:
            answer = ask_question(transcript, question)
            return JsonResponse({"answer": answer})
        return JsonResponse({"transcript": transcript})
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)


@csrf_exempt
@require_POST
def api_file_chat(request):
    try:
        question = request.POST.get('question')
        upload = request.FILES.get('file')
        if not upload or not question:
            return JsonResponse({'error': 'file and question are required'}, status=400)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.name)[1]) as tmp:
            for chunk in upload.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        try:
            out = query_file(tmp_path, question)
            return JsonResponse(out)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)
