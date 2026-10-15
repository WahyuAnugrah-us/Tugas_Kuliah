public class soal1 {
    public static void main(String[] args) {
        int[][] image = {
                { 3, 2, 1, 4, 5 },
                { 4, 3, 2, 0, 1 },
                { 3, 3, 0, 2, 1 },
                { 1, 0, 2, 3, 4 },
                { 0, 1, 4, 1, 2 },
        };
        int[][] filter = {
                { 1, 0, 1 },
                { 0, 1, 1 },
                { 1, 1, 0 }
        };

        int padding = 0;
        int stride = 2;

        int tinggi = image.length;
        int lebar = image[0].length;
        int ukuranFilter = filter.length;

        int tinggiOutput = (tinggi - ukuranFilter + 2 * padding) / stride + 1;
        int lebarOutput = (lebar - ukuranFilter + 2 * padding) / stride + 1;

        int[][] output = new int[tinggiOutput][lebarOutput];

        for (int i = 0; i < tinggiOutput; i++) {
            for (int j = 0; j < lebarOutput; j++) {
                int jumlah = 0;

                for (int k = 0; k < ukuranFilter; k++) {
                    for (int l = 0; l < ukuranFilter; l++) {
                        int barisInput = i * stride + k - padding;
                        int kolomInput = j * stride + l - padding;

                        if (barisInput >= 0
                                && barisInput < tinggi
                                && kolomInput >= 0
                                && kolomInput < lebar) {
                            jumlah += image[barisInput][kolomInput]
                                    * filter[k][l];
                        }
                    }
                }
                output[i][j] = jumlah;
            }
        }

        System.out.println("Output setelah operasi konvolusi :");
        for (int i = 0; i < tinggiOutput; i++) {
            for (int j = 0; j < lebarOutput; j++) {
                System.out.print(output[i][j] + " ");
            }
            System.out.println();
        }
    }
}
