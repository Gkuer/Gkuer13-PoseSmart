from .models import Report
from .serializers import ReportSerializer, ReportListSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

@api_view(['GET', 'POST'])
def reports(request):
    if request.method == 'POST':
        serializer = ReportSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

    elif request.method == 'GET':
        reports_queryset = Report.objects.filter(user=request.user)
        serializer = ReportListSerializer(reports_queryset, many=True)
        return Response(serializer.data)