package deeplearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

public abstract class CNNBaseLayerConvolution extends CNNBaseLayer{
	protected int sizeF;			//フィルタサイズ
	
	protected float[][][][] filter;	//畳み込みフィルタ
	protected float[] bias;			//バイアス
	protected float learnRate; 		//学習率
	
	protected float[][][][] gradient;		//RMSProp適用のための勾配の二乗の総和
	protected float[] gradient_bias;		//バイアス用
	protected float eps = 0.0001f;	//RMSPropでゼロ除算をしないための小さな値
	protected float rmsprop_alpha = 0.9f;	//RMSPropの勾配更新に用いる値
	protected float lambda = 0.001f;		//正則化のための小さな定数
	
	//層情報の保存(CNNクラスのsaveメソッドより呼び出し)
	public void save(BufferedWriter bw) throws IOException{
		int n, k, i, j;
		//畳み込み層の基本情報を書き込み
		bw.write(String.format("%d", layerCode));
		bw.newLine();
		bw.write(String.format("%d", sizeNin));
		bw.newLine();
		bw.write(String.format("%d", iSizeX));
		bw.newLine();
		bw.write(String.format("%d", iSizeY));
		bw.newLine();
		bw.write(String.format("%d", sizeF));
		bw.newLine();
		bw.write(String.format("%d", sizeNout));
		bw.newLine();
		bw.write(String.format("%f", learnRate));
		bw.newLine();
		//フィルタを書き込み
		for( n = 0; n < sizeNin; n++){
			for( k = 0 ; k < sizeNout ; k++){
				for( i = 0 ; i < sizeF ; i++){
					for( j = 0 ; j < sizeF ; j++){
						bw.write(String.format("%f", filter[n][k][i][j]));
						bw.newLine();
					}
				}
			}
		}
		//バイアスを書き込み
		for( k = 0 ; k < sizeNout ; k++){
			bw.write(String.format("%f", bias[k]));
			bw.newLine();
		}
	}
	
	//層情報の読み込み(CNNクラスのloadメソッドより呼び出し)
	public void load(BufferedReader br) throws IOException{
		int n, k, i, j;
		//フィルタを読み込み
		for( n = 0; n < sizeNin; n++){
			for( k = 0 ; k < sizeNout ; k++){
				for( i = 0 ; i < sizeF ; i++){
					for( j = 0 ; j < sizeF ; j++){
						filter[n][k][i][j] = Float.parseFloat(br.readLine());
					}
				}
			}
		}
		//バイアスを読み込み
		for( k = 0 ; k < sizeNout ; k++){
			bias[k] = Float.parseFloat(br.readLine());
		}
	}
}