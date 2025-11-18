const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const result = document.getElementById('result');
const loading = document.getElementById('loading');

imageInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // 이미지 미리보기
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // 예측 시작
    loading.style.display = 'block';
    result.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        loading.style.display = 'none';

        if (data.success) {
            // 정상 예측
            document.getElementById('predictedClass').textContent = data.predicted_class;
            document.getElementById('confidence').textContent = data.confidence;
            result.style.display = 'block';
            result.className = 'result-success';
        } else if (data.is_low_confidence) {
            // 신뢰도 낮음
            document.getElementById('predictedClass').textContent = data.message;
            document.getElementById('confidence').textContent = `(신뢰도: ${data.confidence})`;
            result.style.display = 'block';
            result.className = 'result-warning';
        } else {
            // 기타 오류
            alert('예측 실패: ' + (data.error || data.message));
        }
    } catch (error) {
        loading.style.display = 'none';
        alert('오류가 발생했습니다: ' + error);
    }
});