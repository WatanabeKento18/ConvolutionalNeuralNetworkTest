package deeplearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

public abstract class CNNBaseLayerFullConnect extends CNNBaseLayer{
	protected float[][][][] w;	//重み(入力ノード番号,入力横座標,入力縦座標,出力ノード番号)
	protected float[] bias;	//バイアス
	
	protected float learnRate;	//学習率
	protected float[][][][] gradient;		//RMSProp適用のための勾配の二乗の総和
	protected float eps = 0.0001f;	//RMSPropでゼロ除算をしないための小さな値
	protected float rmsprop_alpha = 0.9f;	//RMSPropの勾配更新に用いる値
	protected float lambda = 0.001f;		//正則化のための小さな定数
	
	//層情報の保存(CNNクラスのsaveメソッドより呼び出し)
	public void save(BufferedWriter bw) throws IOException{
		int i , j , k , l;
		//全結合層の基本情報を書き込み
		bw.write(String.format("%d", layerCode));
		bw.newLine();
		bw.write(String.format("%d", sizeNin));
		bw.newLine();
		bw.write(String.format("%d", iSizeX));
		bw.newLine();
		bw.write(String.format("%d", iSizeY));
		bw.newLine();
		bw.write(String.format("%d", sizeNout));
		bw.newLine();
		bw.write(String.format("%f", learnRate));
		bw.newLine();
		//フィルタを書き込み
		for( k = 0 ; k < sizeNin ; k++){
			for( i = 0 ; i < iSizeX ; i++){
				for( j = 0 ; j < iSizeY ; j++){
					for( l = 0 ; l < sizeNout ; l++){
						bw.write(String.format("%f", w[k][i][j][l]));
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
		int i , j , k , l;
		//重みを読み込み
		for( k = 0 ; k < sizeNin ; k++){
			for( i = 0 ; i < iSizeX ; i++){
				for( j = 0 ; j < iSizeY ; j++){
					for( l = 0 ; l < sizeNout ; l++){
						w[k][i][j][l] = Float.parseFloat(br.readLine());
					}
				}
			}	
		}
		//バイアスを読み込み
		for( k = 0 ; k < sizeNout ; k++){
			bias[k] = Float.parseFloat(br.readLine());
		}
	}
	
	//教師信号を用いた逆伝播
	public abstract float[][][] backPropagation(float[][][] output , float[] label);
}