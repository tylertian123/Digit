package mnist;

import java.lang.reflect.Array;

public final class DatabaseExpander {
	@SafeVarargs
	public static <T> T[] concatArrays(T[]... arrs) {
		int total = 0;
		for(T[] a : arrs)
			total += a.length;
		@SuppressWarnings("unchecked")
		T[] arr = (T[]) Array.newInstance(arrs.getClass().getComponentType().getComponentType(), total);
		int i = 0;
		for(T[] a : arrs) {
			System.arraycopy(a, 0, arr, i, a.length);
			i += a.length;
		}
		return arr;
	}
	
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
	public static MNISTImage[] expandByTranslation2(MNISTImage[] source, int amount) {
		MNISTImage[] out = new MNISTImage[source.length * 2];
		for(int i = 0; i < source.length; i ++) {
			out[i * 2] = translateImage(source[i], amount, amount);
			out[i * 2 + 1] = translateImage(source[i], -amount, -amount);
		}
		return out;
	}
}
