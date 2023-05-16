import unittest
from unittest.mock import MagicMock, patch
from django.test import RequestFactory
from django.http import StreamingHttpResponse
from eyetracking_app import views

class TestViews(unittest.TestCase):

    def setUp(self):
        self.factory = RequestFactory()

    def test_index(self):
        request = self.factory.get('/')
        response = views.index(request)
        self.assertEqual(response.status_code, 200)
    
    def test_get_gaze_direction_out_of_range(self):
        gaze_left = views.get_gaze_direction(-10, 10, 20, 20, 100)
        gaze_right = views.get_gaze_direction(110, 10, 20, 20, 100)
        gaze_center = views.get_gaze_direction(50, -10, 20, 20, 100)

        self.assertEqual(gaze_left, 'left')
        self.assertEqual(gaze_right, 'right')
        self.assertEqual(gaze_center, 'center')

    def test_get_gaze_direction_boundary(self):
        gaze_left = views.get_gaze_direction(0, 0, 20, 20, 100)
        gaze_right = views.get_gaze_direction(100, 0, 20, 20, 100)
        gaze_center = views.get_gaze_direction(50, 0, 20, 20, 100)

        self.assertEqual(gaze_left, 'left')
        self.assertEqual(gaze_right, 'right')
        self.assertEqual(gaze_center, 'center')



    def test_dynamic_stream(self):
        request = self.factory.get('/dynamic_stream/')

        with patch('eyetracking_app.views.cv2.VideoCapture') as mock_video_capture, \
                patch('eyetracking_app.views.gen_frames') as mock_gen_frames:
            # Mock the VideoCapture object and its methods
            mock_video_capture.return_value = MagicMock()
            mock_video_capture.return_value.read.return_value = (True, None)
            mock_video_capture.return_value.release.return_value = None

            # Mock the gen_frames function to return a generator that yields a sample frame
            sample_frame = b'--frame\r\nContent-Type: image/jpeg\r\n\r\nsample_frame\r\n\r\n'
            mock_gen_frames.return_value = (x for x in [sample_frame])

            response = views.dynamic_stream(request)

            # Check if the response is a StreamingHttpResponse
            self.assertIsInstance(response, StreamingHttpResponse)

            # Check if the response contains the expected frame
            self.assertEqual(b''.join(response.streaming_content), sample_frame)

if __name__ == '__main__':
    unittest.main()
