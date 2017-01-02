package deeplearning;

/**
 * 	AtomicFloatÇ∆AtomicDoubleÇçÏÇ¡ÇƒÇ›ÇΩ URL:http://qiita.com/shinido/items/92bc37bb7bf3467654b8
**/

import java.util.concurrent.atomic.AtomicInteger;

public class AtomicFloat {

    private static final int toI(float f){
        return Float.floatToRawIntBits(f);
    }

    private static final float toF(int i){
        return Float.intBitsToFloat(i);
    }

    private final AtomicInteger ai;

    public AtomicFloat(){
        ai = new AtomicInteger();
    }

    public AtomicFloat(float f){
        ai = new AtomicInteger(toI(f));
    }

    public float get() {
        return toF(ai.get());
    }

    public float getAndAdd(float delta) {
        int i;
        do {
            i = ai.get();
        } while(!ai.compareAndSet(i, toI(toF(i) + delta)));

        return toF(i);
    }

    public float addAndGet(float delta){
        int i;
        do {
            i = ai.get();
        } while(!ai.compareAndSet(i, toI(toF(i) + delta)));
        return get();
    }

    public boolean compareAndSet(float expect, float update) {
        return ai.compareAndSet(toI(expect), toI(update));
    }

    public void set(float newValue) {
        ai.set(toI(newValue));
    }

    public float getAndSet(float newValue) {
        return toF(ai.getAndSet(toI(newValue)));
    }

    public boolean weakCompareAndSet(float expect, float update) {
        return ai.weakCompareAndSet(toI(expect), toI(update));
    }

    public void lazySet(float newValue) {
        ai.lazySet(toI(newValue));
    }

}