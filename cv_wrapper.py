import numpy as np
import cv2 as cv
import imageio


class CVWrapper:
	@staticmethod
	def instantiate_window(window_title):
		cv.namedWindow(window_title)

	@staticmethod
	def connect_window_mouse_callback(window_title, on_mouse_event=lambda event, x, y, flags: event == cv.EVENT_LBUTTONDOWN, param=None):
		cv.setMouseCallback(window_title, on_mouse_event, param)

	@staticmethod
	def write_on_window(window_title, image):
		cv.imshow(window_title, image)

	@staticmethod
	def image_sized(image, size=(1920, 1080)):
		return cv.resize(image, size)

	@staticmethod
	def image_scaled(image, scale_x=1.0, scale_y=1.0):
		return cv.resize(image, (0, 0), fx=scale_x, fy=scale_y)

	@staticmethod
	def image_contrast_stretched(image, is_grayscale=False, contrast=1.6, brightness=-40):
		img_gray = image if is_grayscale else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		img_tran = (contrast * img_gray) + brightness
		img_tran[img_tran < 0] = 0
		img_tran[img_tran > 255] = 255
		img_tran = img_tran.astype(np.uint8)

		return img_tran

	@staticmethod
	def image_alpha_blended(image_1, image_2, alpha=0.5):
		img_blend = alpha * image_1 + (1 - alpha) * image_2
		img_blend = img_blend.astype(np.uint8)
		# img_blend = cv.addWeighted(image_1, 0.25, image_2, 0.75, 0)

		return img_blend

	@staticmethod
	def image_difference(image_1, image_2):
		img_diff = np.abs(image_1.astype(np.int32) - image_2).astype(np.uint8)
		# img_diff = cv.absdiff(image_1, image_2)

		return img_diff

	@staticmethod
	def image_binary_thresholded(image, is_grayscale=False, threshold=127, threshold_type=cv.THRESH_BINARY_INV):
		img_gray = image if is_grayscale else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		_, img_binary = cv.threshold(img_gray, threshold, 255, threshold_type)

		return img_binary

	@staticmethod
	def image_text_drawn(image, text, font=cv.FONT_HERSHEY_DUPLEX, font_scale=1.0, color=(255, 255, 255), thickness=1, anchor=(0.5, 0.5), offset=(0, 0)):
		h, w, *_ = image.shape
		origin = (int(w * anchor[0]) + offset[0], int(h * anchor[1]) + offset[1])

		return cv.putText(np.copy(image), text, origin, font, font_scale, color=color, thickness=thickness)

	@staticmethod
	def image_circle_drawn(image, radius, color=(255, 255, 255), thickness=-1, anchor=(0.5, 0.5), offset=(0, 0)):
		h, w, *_ = image.shape
		center = (int(w * anchor[0]) + offset[0], int(h * anchor[1]) + offset[1])

		return cv.circle(np.copy(image), center, radius, color=color, thickness=thickness)

	@staticmethod
	def image_chessboard_corners_drawn(image, board_pattern=(7, 7)):
		complete, points = cv.findChessboardCorners(image, board_pattern)

		return cv.drawChessboardCorners(np.copy(image), board_pattern, points, complete)

	@staticmethod
	def edges_with_canny(image, is_grayscale=False, lower_threshold=30, upper_threshold=100):
		img_gray = image if is_grayscale else cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		img_gray_blured = cv.medianBlur(img_gray, 5)
		edges = cv.Canny(img_gray_blured, threshold1=lower_threshold, threshold2=upper_threshold)

		return edges

	@staticmethod
	def draw_text_on_image(image, text, font=cv.FONT_HERSHEY_DUPLEX, font_scale=1.0, color=(255, 255, 255), thickness=1, anchor=(0.5, 0.5), offset=(0, 0)):
		h, w, *_ = image.shape
		origin = (int(w * anchor[0]) + offset[0], int(h * anchor[1]) + offset[1])
		cv.putText(image, text, origin, font, font_scale, color=color, thickness=thickness)

	@staticmethod
	def draw_circle_on_image(image, radius, color=(255, 255, 255), thickness=-1, anchor=(0.5, 0.5), offset=(0, 0)):
		h, w, *_ = image.shape
		center = (int(w * anchor[0]) + offset[0], int(h * anchor[1]) + offset[1])
		cv.circle(image, center, radius, color=color, thickness=thickness)

	@staticmethod
	def draw_chessboard_corners(image, board_pattern=(7, 7)):
		complete, points = cv.findChessboardCorners(image, board_pattern)
		cv.drawChessboardCorners(image, board_pattern, points, complete)


class ImageWrapper:
	@staticmethod
	def write_gif(image_path, frames, fps):
		imageio.mimsave(image_path, frames, fps=fps)

	@staticmethod
	def write_image(image_path, image):
		imageio.imwrite(image_path, image)
		# cv.imwrite(image_path, image)


class VideoWrapper:
	def __init__(self):
		# Video Reader
		self.vreader = None
		self.read_fps = 0 # Hz
		self.read_delta = 0 # Millisec

		# Video Writer
		self.vwriter = None
		self.write_path = ""
		self.write_codec = None

		# Extraction from Video
		self.image_background = None
		self.bg_frame_count = 0

		self.snapshots = []

	def setup_reader_with_index(self, video_index=0): # Open a camera to read
		self.vreader = cv.VideoCapture(video_index)
		assert self.vreader.isOpened(), "Read error: Failed to take a camera!"

		self.read_fps = self.vreader.get(cv.CAP_PROP_FPS)
		self.read_delta = int(1000 * (1 / self.read_fps))

	def setup_reader_with_file(self, video_file): # Open a video to read
		self.vreader = cv.VideoCapture(video_file)
		assert self.vreader.isOpened(), "Read error: Failed to take a video!"

		self.read_fps = self.vreader.get(cv.CAP_PROP_FPS)
		self.read_delta = int(1000 * (1 / self.read_fps))

	def setup_writer_with_path(self, video_path="output.mp4", codec="mp4v"): # Get ready to write a video
		self.vwriter = cv.VideoWriter()
		self.write_path = video_path
		self.write_codec = cv.VideoWriter_fourcc(*codec)

	def read_frame(self):
		valid, frame = self.vreader.read()
		if not valid:
			return None

		return frame

	def write_frame(self, frame, fps=30):
		if not self.vwriter.isOpened():
			h, w, *_ = frame.shape
			opened = self.vwriter.open(self.write_path, self.write_codec, fps, (w, h))
			assert opened, "Write error: Failed to open a video!"

		self.vwriter.write(frame)

	def add_frame_as_background(self, frame):
		if self.image_background is None:
			self.image_background = np.zeros_like(frame, dtype=np.float64)

		self.image_background += frame.astype(np.float64)
		self.bg_frame_count += 1

	def get_image_background(self):
		return (self.image_background / self.bg_frame_count).astype(np.uint8)

	def pause(self, current_frame, resume_key=ord(' '), snapshot_key=ord('\r')):
		while True:
			key = cv.waitKey()
			if key == resume_key:
				break
			elif key == snapshot_key:
				self.snapshots.append(current_frame)
				break

	def close(self):
		if self.vwriter is not None:
			self.vwriter.release()


def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
	# Find 2D corner points from given images
	img_points = []
	for img in images:
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		complete, pts = cv.findChessboardCorners(gray, board_pattern)
		if complete:
			img_points.append(pts)
	assert len(img_points) > 0

	# Prepare 3D points of the chess board
	obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
	obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

	# Calibrate the camera
	return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)



if __name__ == "__main__":
	video_file = "res/chessboard_2.mp4"
	board_pattern = (7, 7)
	board_cellsize = 0.025

	# Camera calibration (part 1)
	window_title = "Selecting Images"
	vreader = VideoWrapper()
	vreader.setup_reader_with_file(video_file)

	while True:
		frame = vreader.read_frame()
		if frame is None:
			break

		# Process the frame
		frame_to_show = CVWrapper.image_text_drawn(frame, f'NSelect: {len(vreader.snapshots)}', anchor=(0, 0), offset=(10, 25))
		CVWrapper.write_on_window(window_title, frame_to_show)

		# Process the keyboard input
		key = cv.waitKey(vreader.read_delta)
		if key == 27: # ESC
			break
		elif key == ord(' '):
			CVWrapper.write_on_window(window_title, CVWrapper.image_chessboard_corners_drawn(frame_to_show, board_pattern))
			vreader.pause(frame)
	cv.destroyAllWindows()

	# Obtain calibration results
	img_select = vreader.snapshots
	assert len(img_select) > 0, 'There is no selected images!'
	rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)\

	# Print calibration results
	print('## Camera Calibration Results')
	print(f'* The number of selected images = {len(img_select)}')
	print(f'* RMS error = {rms}')
	print(f'* Camera matrix (K) = \n{K}')
	print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

	# Triangle rendering in AR (part 2)
	vwrapper = VideoWrapper()
	vwrapper.setup_reader_with_file(video_file)
	vwrapper.setup_writer_with_path("res/output.mp4", "X264")

	# Prepare a 3D box for simple AR
	board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
	box_lower = board_cellsize * np.array([[3, 2,  0], [5, 2,  0], [4, 3,  0]])
	box_upper = board_cellsize * np.array([[3, 2, -1], [5, 2, -1], [4, 3, -1]])

	# Prepare 3D points on a chessboard
	obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

	while True:
		frame = vwrapper.read_frame()
		if frame is None:
			break

		# Estimate the camera pose
		img = np.copy(frame)
		success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
		if success:
			ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

			# Draw the box on the image
			line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
			line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
			cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
			cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)
			for b, t in zip(line_lower, line_upper):
				cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

			# Print the camera position
			R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
			p = (-R.T @ tvec).flatten()
			info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
			cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

		# Show the image
		CVWrapper.write_on_window("Triangle Rendering", img)
		vwrapper.write_frame(img, fps=vreader.read_fps)

		# Process the keyboard input
		key = cv.waitKey(vreader.read_delta)
		if key == 27: # ESC
			break
		elif key == ord(' '):
			vwrapper.pause(img)
		# elif key == ord('\t'):
		# 	pass
		# elif key == ord('+') or key == ord('='):
		# 	pass
		# elif key == ord('-') or key == ord('_'):
		# 	pass

	idx = 0
	for snapshot in vwrapper.snapshots:
		ImageWrapper.write_image(f"res/pose_{idx}.jpg", snapshot)
		idx += 1
	vwrapper.close()
	cv.destroyAllWindows()