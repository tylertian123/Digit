package mnist;

public final class DatabaseExpander {
	static int constrain(int val, int upper, int lower) {
		int a = val < upper ? val : upper;
		return a > lower ? a : lower;
	}
	static MNISTImage translateImage(MNISTImage src, int xa, int ya) {
		MNISTImage img = new MNISTImage((int) src.getClassification());
		for(int x = 0; x < MNISTImage.SIZE; x ++) {
			for(int y = 0; y < MNISTImage.SIZE; y ++) {
				img.set(constrain(x + xa, MNISTImage.SIZE - 1, 0), constrain(y + ya, MNISTImage.SIZE - 1, 0), src.get(x, y));
			}
		}
		return img;
	}
	public static MNISTImage[] expandByTranslation(MNISTImage[] source, int amount) {
		MNISTImage[] out = new MNISTImage[source.length * 4];
		for(int i = 0; i < source.length; i ++) {
			out[i * 4] = translateImage(source[i], 0, -amount);
			out[i * 4 + 1] = translateImage(source[i], amount, 0);
			out[i * 4 + 2] = translateImage(source[i], 0, amount);
			out[i * 4 + 3] = translateImage(source[i], -amount, 0);
		}
		return out;
	}
}
