import java.util.ArrayList;

public class soal2 {
    public static void main(String[] args) {
        int[][] image = {
                { -3, -2, -1, 4, 5 },
                { -4, -3, 2, 0, 1 },
                { -3, 3, 0, 2, 1 },
                { 1, 0, 2, -3, -4 },
                { 0, 1, -4, -1, 2 }
        };

        for (int[] image1 : image) {
            for (int j = 0; j < image1.length; j++) {
                if (image1[j] <= 0) {
                    image1[j] = 0;
                }
            }
        }

        System.out.println("Image setelah operasi ReLU :");
        for (int[] image1 : image) {
            for (int j = 0; j < image1.length; j++) {
                System.out.print(image1[j] + " ");
            }
            System.out.println();
        }

        ArrayList<Integer> flatten = new ArrayList<>();
        for (int[] image1 : image) {
            for (int j = 0; j < image1.length; j++) {
                flatten.add(image1[j]);
            }
        }

        System.out.println("Flattened array:");
        System.out.println(flatten);

    }
}
