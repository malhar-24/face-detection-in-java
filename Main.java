import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import java.io.File;

public class Main {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        // Load the face detection model
        String faceCascadePath = "C:\\Users\\HP\\IdeaProjects\\FaceDetection\\.idea\\haarcascade_frontalface_default.xml";
        CascadeClassifier faceDetector = new CascadeClassifier(faceCascadePath);

        if (faceDetector.empty()) {
            System.out.println("Error loading face cascade file.");
            return;
        }

        // Define the archive folder path containing subfolders of faces
        String archiveFolderPath = "C:\\Users\\HP\\IdeaProjects\\FaceDetection\\archive";

        // Open the default camera
        VideoCapture camera = new VideoCapture(0);

        // Create a matrix to store the video frame
        Mat frame = new Mat();

        // Continuously capture video from the camera
        while (true) {
            // Read the camera frame into the matrix
            if (camera.read(frame)) {
                // Convert the frame to grayscale (required for face detection)
                Mat grayscaleFrame = new Mat();
                Imgproc.cvtColor(frame, grayscaleFrame, Imgproc.COLOR_BGR2GRAY);

                // Detect faces in the camera feed
                MatOfRect faceDetections = new MatOfRect();
                faceDetector.detectMultiScale(grayscaleFrame, faceDetections);

                for (Rect rect : faceDetections.toArray()) {
                    // Crop the detected face
                    Mat detectedFace = new Mat(grayscaleFrame, rect);
                    Imgproc.resize(detectedFace, detectedFace, new Size(100, 100)); // Resize to compare with stored faces

                    // Apply histogram equalization for better contrast
                    Imgproc.equalizeHist(detectedFace, detectedFace);

                    // Compare with stored faces in the archive
                    String matchedName = compareFaces(detectedFace, archiveFolderPath);
                    if (matchedName != null) {
                        // Display the name if a match is found
                        Imgproc.putText(frame, matchedName, new Point(rect.x, rect.y - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);
                    } else {
                        Imgproc.putText(frame, "Unknown", new Point(rect.x, rect.y - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 0, 255), 2);
                    }

                    // Draw rectangle around detected face
                    Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                            new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(0, 255, 0), 3);
                }

                // Display the frame with detected faces and names
                HighGui.imshow("Camera Face Recognition", frame);

                // Break the loop if the 'q' key is pressed
                if (HighGui.waitKey(30) == 'q') {
                    break;
                }
            }
        }

        // Release the camera and close windows
        camera.release();
        HighGui.destroyAllWindows();
    }

    // Method to compare detected face with stored faces in archive folder
    public static String compareFaces(Mat detectedFace, String archiveFolderPath) {
        File archiveFolder = new File(archiveFolderPath);
        if (!archiveFolder.exists() || !archiveFolder.isDirectory()) {
            System.out.println("Archive folder not found.");
            return null;
        }

        // Loop through each subfolder (actor/actress)
        for (File subFolder : archiveFolder.listFiles()) {
            if (subFolder.isDirectory()) {
                String actorName = subFolder.getName();

                // Loop through each image in the subfolder
                for (File imageFile : subFolder.listFiles()) {
                    if (imageFile.isFile() && imageFile.getName().endsWith(".jpg")) {
                        // Load the stored face image
                        Mat storedFace = Imgcodecs.imread(imageFile.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
                        if (!storedFace.empty()) {
                            Imgproc.resize(storedFace, storedFace, new Size(100, 100)); // Resize to same size as detected face

                            // Apply histogram equalization for better contrast
                            Imgproc.equalizeHist(storedFace, storedFace);

                            // Ensure both images are of the same type and size
                            if (storedFace.size().equals(detectedFace.size()) && storedFace.type() == detectedFace.type()) {
                                // Calculate the similarity using absolute difference
                                Mat diff = new Mat();
                                Core.absdiff(detectedFace, storedFace, diff);

                                // Sum the differences (lower values indicate more similarity)
                                double totalDiff = Core.sumElems(diff).val[0];

                                if (totalDiff < 400000) { // Adjust threshold based on your testing
                                    return actorName;
                                }
                            }
                        }
                    }
                }
            }
        }
        return null; // No match found
    }
}
