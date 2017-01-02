package deeplearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

/**
 * 
 * 畳み込みニューラルネットワーク：活性化層
 *
 */
public class CNNActivation extends CNNBaseLayer{
	boolean activationMap[][][];		//活性化された座標を保存
	
	CNNActivation(int sizeN , int sizeX , int sizeY){
		this.layerCode = 1021;
		this.sizeNin = this.sizeNout = sizeN;
		this.iSizeX = this.oSizeX = sizeX;
		this.iSizeY = this.oSizeY = sizeY;
		activationMap = new boolean[sizeN][sizeX][sizeY];		
	}
	
	
	//順伝播(活性化関数:ReLU)
	public float[][][] propagation(float[][][] input){
		int n , i , j;
		float[][][] output = new float[sizeNin][iSizeX][iSizeY];
		for( n = 0 ; n < sizeNin ; n++){
			for( i = 0 ; i < iSizeX ; i++){
				for( j = 0 ; j < iSizeY ; j++){
					if(input[n][i][j] > 0){
						output[n][i][j] = input[n][i][j];
						activationMap[n][i][j] = true;
					}else{
						output[n][i][j] = 0;
						activationMap[n][i][j] = false;
					}
				}
			}
		}
		return output;
	}
	
	//逆伝播
	//引数:[ノード番号][入力X軸][入力Y軸]
	public float[][][] backPropagation(float[][][] error){
		int n , i , j;
		float[][][] nextError = new float[sizeNin][iSizeX][iSizeY];
		for( n = 0 ; n < sizeNin ; n++){
			for( i = 0 ; i < iSizeX ; i++){
				for( j = 0 ; j < iSizeY ; j++){
					if(activationMap[n][i][j]){
						nextError[n][i][j] = error[n][i][j];
					}else{
						nextError[n][i][j] = 0;
					}
					//System.out.println("NEXT ERR ACTV:"+nextError[n][i][j] + " i:"+i+" j:"+j);
				}
			}
		}
		return nextError;
	}
	
	public void training(){		
	}
	
	//層情報の保存(CNNクラスのsaveメソッドより呼び出し)
	public void save(BufferedWriter bw) throws IOException{
		//活性化層の基本情報を書き込み
		bw.write(String.format("%d", layerCode));
		bw.newLine();
		bw.write(String.format("%d", sizeNin));
		bw.newLine();
		bw.write(String.format("%d", iSizeX));
		bw.newLine();
		bw.write(String.format("%d", iSizeY));
		bw.newLine();
	}
	
	//層情報の読み込み(CNNクラスのloadメソッドより呼び出し)
	public void load(BufferedReader br) throws IOException{
	
	}
		
	//テスト用
	public static void main(String[] args){
		float[][][] image = {{{-0.5f , 0.0f , 0.0f , 0.0f , 0.0f , 0.0f , 0.0f , 0.1f , 0.9f},
							{0.9f , 0.5f , 0.0f , 0.1f , 0.0f , 0.0f , 0.9f , 0.9f , 0.9f},
							{-0.9f , 0.9f , 0.1f , 0.0f , 0.0f , 0.1f , 0.9f , 0.1f , 0.0f},
							{0.1f , 0.9f , 0.1f , -0.1f , 0.0f , 0.1f , 0.9f , 0.1f , 0.0f},
							{-0.1f , 0.9f , 0.9f , 0.1f , 0.0f , 0.9f , 0.9f , 0.9f , 0.1f},
							{0.0f , -0.1f , 0.9f , -0.9f , 0.9f , 0.1f , 0.2f , 0.9f , 0.1f}}};
		System.out.println("活性化処理前");
		for(int j = 0 ; j < 6 ; j++){
			for(int i = 0 ; i < 9 ; i++){
				System.out.print(image[0][j][i] + " ");
			}
			System.out.println();
		}
		System.out.println("活性化処理後");
		CNNActivation actv = new CNNActivation(1,6,9);
		float[][][] output = actv.propagation(image);
		for(int i = 0 ; i < output[0].length ; i++){
			for(int j = 0 ; j < output[0][0].length ; j++){
				System.out.print(String.format("%.1f",output[0][i][j]) + " ");
			}
			System.out.println();
		}
	}
}