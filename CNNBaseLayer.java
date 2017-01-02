package deeplearning;

import java.io.*;

/**
 * 		CNN層の基本となる構造
**/

public abstract class CNNBaseLayer{
	protected int layerCode;			//層の識別番号
	protected int sizeNin;				//入力ノード数
	protected int iSizeX , iSizeY;		//入力サイズ
	protected int sizeNout;				//出力ノード数
	protected int oSizeX , oSizeY;		//出力サイズ
	
	protected boolean useMinibatch;		//ミニバッチ学習を行う場合true
	protected float[][][] beforeInput;				//学習用入力データ
	
	public abstract float[][][] propagation(float[][][] input);
	public abstract float[][][] backPropagation(float[][][] error);
	public abstract void training();
	public abstract void save(BufferedWriter bw) throws IOException;
	public abstract void load(BufferedReader br) throws IOException;

	//ミニバッチ学習設定
	public void initMinibatchSetting(int minibatchSize){
		useMinibatch = true;
	}
	
	//入力サイズの配列を取得
	public float[][][] getInputClone(){
		return new float[sizeNin][iSizeX][iSizeY];
	}
	
	//学習用の入力データを設定
	public void setInput(float[][][] input){
		beforeInput = input;
	}
	
	//出力側のサイズをまとめて取得
	public int[] getOutputSizeList(){
		int[] sizeList = new int[3];
		sizeList[0] = sizeNout;
		sizeList[1] = oSizeX;
		sizeList[2] = oSizeY;
		return sizeList;
	}
}