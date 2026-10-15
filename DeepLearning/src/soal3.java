public class soal3 {
    public static void main(String[] args) {
        int[][] image = {
                { 3, 2, 1, 4, 5 },
                { 4, 3, 2, 0, 1 },
                { 3, 3, 0, 2, 1 },
                { 1, 0, 2, 3, 4 },
                { 0, 1, 4, 1, 2 },
        };
        int ukuranFilter = 2;
        int stride = 2;

        int tinggiOutput = (image.length - ukuranFilter) / stride + 1;
        int lebarOutput = (image[0].length - ukuranFilter) / stride + 1;

        int[][] output = new int[tinggiOutput][lebarOutput];

        for (int i = 0; i < tinggiOutput; i++) {
            for (int j = 0; j < lebarOutput; j++) {
                int maksimum = Integer.MIN_VALUE;

                for (int k = 0; k < ukuranFilter; k++) {
                    for (int l = 0; l < ukuranFilter; l++) {
                        int nilai = image[i * stride + k][j * stride + l];

                        if (nilai > maksimum) {
                            maksimum = nilai;
                        }
                    }
                }

                output[i][j] = maksimum;
            }
        }

        System.out.println("Output setelah operasi max pooling :");
        for (int i = 0; i < tinggiOutput; i++) {
            for (int j = 0; j < lebarOutput; j++) {
                System.out.print(output[i][j] + " ");
            }
            System.out.println();
        }
    }
}
