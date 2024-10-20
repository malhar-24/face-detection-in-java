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

        String faceCascadePath = "C:\\Users\\HP\\IdeaProjects\\FaceDetection\\.idea\\haarcascade_frontalface_default.xml";
        CascadeClassifier faceDetector = new CascadeClassifier(faceCascadePath);

        if (faceDetector.empty()) {
            System.out.println("Error loading face cascade file.");
            return;
        }

        String archiveFolderPath = "C:\\Users\\HP\\IdeaProjects\\FaceDetection\\archive";

        VideoCapture camera = new VideoCapture(0);

        Mat frame = new Mat();

        while (true) {
            if (camera.read(frame)) {
                Mat grayscaleFrame = new Mat();
                Imgproc.cvtColor(frame, grayscaleFrame, Imgproc.COLOR_BGR2GRAY);

                MatOfRect faceDetections = new MatOfRect();
                faceDetector.detectMultiScale(grayscaleFrame, faceDetections);

                for (Rect rect : faceDetections.toArray()) {
                    Mat detectedFace = new Mat(grayscaleFrame, rect);
                    Imgproc.resize(detectedFace, detectedFace, new Size(100, 100)); // Resize to compare with stored faces

                    Imgproc.equalizeHist(detectedFace, detectedFace);

                    String matchedName = compareFaces(detectedFace, archiveFolderPath);
                    if (matchedName != null) {
                        Imgproc.putText(frame, matchedName, new Point(rect.x, rect.y - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);
                    } else {
                        Imgproc.putText(frame, "Unknown", new Point(rect.x, rect.y - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 0, 255), 2);
                    }

                    Imgproc.rectangle(frame, new Point(rect.x, rect.y),
                            new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(0, 255, 0), 3);
                }

                HighGui.imshow("Camera Face Recognition", frame);

                if (HighGui.waitKey(30) == 'q') {
                    break;
                }
            }
        }

        camera.release();
        HighGui.destroyAllWindows();
    }

    public static String compareFaces(Mat detectedFace, String archiveFolderPath) {
        File archiveFolder = new File(archiveFolderPath);
        if (!archiveFolder.exists() || !archiveFolder.isDirectory()) {
            System.out.println("Archive folder not found.");
            return null;
        }

        for (File subFolder : archiveFolder.listFiles()) {
            if (subFolder.isDirectory()) {
                String actorName = subFolder.getName();

                for (File imageFile : subFolder.listFiles()) {
                    if (imageFile.isFile() && imageFile.getName().endsWith(".jpg")) {
                        Mat storedFace = Imgcodecs.imread(imageFile.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
                        if (!storedFace.empty()) {
                            Imgproc.resize(storedFace, storedFace, new Size(100, 100)); // Resize to same size as detected face

                            Imgproc.equalizeHist(storedFace, storedFace);

                            if (storedFace.size().equals(detectedFace.size()) && storedFace.type() == detectedFace.type()) {
                                Mat diff = new Mat();
                                Core.absdiff(detectedFace, storedFace, diff);

                                double totalDiff = Core.sumElems(diff).val[0];

                                if (totalDiff < 400000) {
                                    return actorName;
                                }
                            }
                        }
                    }
                }
            }
        }
        return null;
    }
}
