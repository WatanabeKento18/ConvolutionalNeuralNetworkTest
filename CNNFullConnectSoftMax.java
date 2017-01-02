package deeplearning;

import java.util.*;

/**
 * 
 * 畳み込みニューラルネットワーク：全結合層(活性化関数:SoftMax)
 *
 */

public class CNNFullConnectSoftMax extends CNNBaseLayerFullConnect{
	private float[][][] beforeError;	//直前の誤差
	
	//sizeNin:入力ノード数 sizeNout:出力ノード数 iSizeX:入力データ横サイズ iSizeY:入力データ縦サイズ
	public CNNFullConnectSoftMax(int sizeNin , int iSizeX , int iSizeY, int sizeNout ){
		this.layerCode = 1041;
		this.sizeNin = sizeNin;
		this.sizeNout = sizeNout;
		this.iSizeX = iSizeX;
		this.iSizeY = iSizeY;
		this.oSizeX = this.oSizeY = 1;
		
		//乱数生成器
		Random rnd = new Random();
		
		//初期化
		w = new float[sizeNin][iSizeX][iSizeY][sizeNout];
		for(int ni = 0 ; ni < sizeNin ; ni++){
			for(int i = 0 ; i < iSizeX ; i++){
				for(int j = 0 ; j < iSizeY ; j++){
					for(int no = 0 ; no < sizeNout ; no++){
						w[ni][i][j][no] = rnd.nextFloat() - 0.5f;
					}
				}				
			}			
		}
		bias = new float[sizeNout];
		for(int no = 0 ; no < sizeNout ; no++){
			bias[no] = 1.0f;
		}
		gradient = new float[sizeNin][iSizeX][iSizeY][sizeNout];
		
		learnRate = 0.001f;
	}
	
	//学習率設定
	public void setLearningRate(float f){
		if(f < 0) f = 0.0f;
		else if(f > 1.0f) f = 1.0f;
		learnRate = f;
	}
	
	//順伝播（引数　input[入力ノード番号][X][Y]：入力データ　返り値　output[出力ノード番号]：出力データ）
	public float[][][] propagation(float[][][] input){

		//出力変数生成
		float[][][] output = new float[sizeNout][1][1];
		
		//出力を計算
		int ni , no , i , j ;
		for(no = 0 ; no < sizeNout ; no++){
			output[no][0][0] = 0;
			for(ni = 0 ; ni < sizeNin ; ni++){
				//重み計算
				for(i = 0 ; i < iSizeX ; i++){
					for(j = 0 ; j < iSizeY ; j++){
						output[no][0][0] += input[ni][i][j] * w[ni][i][j][no];
					}
				}
			}
			output[no][0][0] += bias[no];	//バイアス加算
			//System.out.println("PROP:"+output[no]);
		}
		output = softmax(output);
		
		//学習用に入力を記憶
		if(!useMinibatch){
			beforeInput = input;
		}
		return output;
	}
	
	//オリジナル
	//逆伝播(引数 output[ノード番号]:出力層側のデータ input[]:入力層側のデータ label[ノード番号]　返り値 error[ノード番号][X][Y] この層の重みを伝った誤差)
	public float[][][] backPropagation(float[][][] output , float[] label){
		float[][][] error = new float[sizeNin][iSizeX][iSizeY];	//誤差記憶用
		float[] d = new float[sizeNout];
		
		int no, ni, i, j;
		float eperror;	//学習率乗算後の誤差(一時記憶用)
		//学習
		for(no = 0 ; no < sizeNout ; no++){
			d[no] = output[no][0][0] - label[no];	//出力とラベルデータの誤差を算出
			//重み用ループ
			//gradient[no] = rmsprop_alpha * gradient[no] + (1 - rmsprop_alpha)*d[no]*d[no];
			eperror = learnRate * d[no];
			for(ni = 0 ; ni < sizeNin ; ni++){
				for(i = 0 ; i < iSizeX ; i++){
					for(j = 0 ; j < iSizeY ; j++){
						error[ni][i][j] += d[no] * w[ni][i][j][no];	//伝播させる誤差を算出
						w[ni][i][j][no] -= eperror * beforeInput[ni][i][j];	//重みを更新
					}
				}
			}
			
			//バイアスを更新
			bias[no] -= eperror;
		}
		return error;
	}

	//オリジナル
	//逆伝播(引数 beforeError[出力ノード番号]:前層の誤差 input[]:入力層側のデータ label　返り値 error[ノード番号][X][Y] この層の重みを伝った誤差)
	public float[][][] backPropagation_(float[][][] beforeError){
		float[][][] error = new float[sizeNin][iSizeX][iSizeY];	//誤差記憶用
		
		int no , ni , i , j;
		float eperror;	//学習率乗算後の誤差(一時記憶用)
		//学習
		for(no = 0 ; no < sizeNout ; no++){
			//重み用ループ
			eperror = learnRate * beforeError[no][0][0];
			for(ni = 0 ; ni < sizeNin ; ni++){
				for(i = 0 ; i < iSizeX ; i++){
					for(j = 0 ; j < iSizeY ; j++){
						error[ni][i][j] += beforeError[no][0][0] * w[ni][i][j][no];	//伝播させる誤差を算出
						w[ni][i][j][no] -= eperror * beforeInput[ni][i][j];	//重みを更新
					}
				}
			}
			
			//バイアスを更新
			bias[no] -= eperror;
		}
		return error;
	}

	// オリジナル(ミニバッチ用)
	// 逆伝播(引数 beforeError[出力ノード番号]:前層の誤差 input[]:入力層側のデータ label　返り値 error[ノード番号][X][Y] この層の重みを伝った誤差)
	public float[][][] backPropagation(float[][][] beforeError) {
		float[][][] error = new float[sizeNin][iSizeX][iSizeY]; // 誤差記憶用

		int no, ni, i, j;
		// 学習
		for (no = 0; no < sizeNout; no++) {
			// 重み用ループ
			for (ni = 0; ni < sizeNin; ni++) {
				for (i = 0; i < iSizeX; i++) {
					for (j = 0; j < iSizeY; j++) {
						error[ni][i][j] += beforeError[no][0][0]*w[ni][i][j][no]; // 伝播させる誤差を算出
						//if(Float.isNaN(error[ni][i][j])) System.out.println("FC ERR NAN:"+ni+":"+i+":"+j+":"+no);
					}
				}
			}
			
			//バイアスを更新
			bias[no] -= learnRate * beforeError[no][0][0];
		}
		if(Float.isNaN(beforeError[0][0][0])) System.out.println("FC1 EXCEPTION!!");
		if(Float.isNaN(error[0][0][0])) System.out.println("FC2 EXCEPTION!!");
		this.beforeError = beforeError;
		return error;
	}
	
	//パラメータ調整
	public void training(){
		int no , ni , i , j;
		float totalError;	//一時記憶用
		final float rmsprop_alpha2 = 1 - rmsprop_alpha;
		//学習
		for(no = 0 ; no < sizeNout ; no++){
			//重み用ループ
			for(ni = 0 ; ni < sizeNin ; ni++){
				for(i = 0 ; i < iSizeX ; i++){
					for(j = 0 ; j < iSizeY ; j++){
						totalError = beforeError[no][0][0] * beforeInput[ni][i][j];
						gradient[ni][i][j][no] = rmsprop_alpha*gradient[ni][i][j][no] + rmsprop_alpha2*totalError*totalError;
						w[ni][i][j][no] -= learnRate*totalError/(Math.sqrt(gradient[ni][i][j][no])+eps);	//重みを更新
						//if(Float.isNaN(totalError)) System.out.println("FC NAN:"+ni+":"+i+":"+j+":"+no);
					}
				}
			}
		}
	}
	
	//重みを更新(引数 input:入力)
	public void updateParameter(float[][][] input){
		
	}
	
	//softmax関数
	private float[][][] softmax(float[][][] input){
		int length = input.length;	//配列長
		float[][][] output = new float[length][1][1];
		float max = input[0][0][0];	//最大値記憶用
		float sum = 0;	//合計値記憶用
		
		//無限大回避のための前処理
		for(int i = 1 ; i < length ; i++){
			if(max < input[i][0][0]){
				max = input[i][0][0];
			}
			//System.out.println("MAX INPUT:"+max+" "+input[i]);
		}
		
		//合計値を算出
		for(int i = 0 ; i < length ; i++){
			input[i][0][0] -= max;
			sum += Math.exp(input[i][0][0]);
		}

		for(int i = 0 ; i < length ; i++){
			output[i][0][0] = (float)Math.exp(input[i][0][0])/sum;
		}
		if(Float.isNaN(output[0][0][0])){
			System.out.println("なぜかエラーが出ました : "+sum);
			for(int no = 0 ; no < sizeNout ; no++){
				System.out.println("["+no+"]:"+input[no][0][0]);
			}
			//System.exit(0);
		}
		return output;
	}
	
	//テスト用
	public static void main(String[] args){
		float[][][] train ={
				{	{1.0f , 1.0f , 0.0f},
					{1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f},},
				{	{0.0f , 1.0f , 0.0f},
					{1.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 0.0f},},
				{	{0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 1.0f},
					{0.0f , 1.0f , 1.0f},},
				{	{0.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 1.0f},
					{0.0f , 1.0f , 0.0f},}
		};
		
		float[][] label = {
				{1.0f,0.0f},
				{1.0f,0.0f},
				{0.0f,1.0f},
				{0.0f,1.0f},
		};
		
		float[][][] test ={
				{	{1.0f , 1.0f , 0.0f},
					{1.0f , 1.0f , 0.0f},
					{0.0f , 0.0f , 0.0f},},
				{	{0.0f , 1.0f , 0.0f},
					{1.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 0.0f},},
				{	{0.0f , 0.0f , 0.0f},
					{0.0f , 1.0f , 1.0f},
					{0.0f , 1.0f , 1.0f},},
				{	{0.0f , 0.0f , 0.0f},
					{0.0f , 0.0f , 1.0f},
					{0.0f , 1.0f , 0.0f},}
		};
		
		CNNFullConnectSoftMax fcl = new CNNFullConnectSoftMax(1,3,3,2);
		for(int epoch = 0 ; epoch < 1000 ; epoch++){
			for(int data = 0 ; data < 4 ; data++){
				float[][][] input = new float[1][3][3];
				for(int i = 0 ; i < 3 ; i++){
					for(int j = 0 ; j < 3 ; j++){
						input[0][i][j] = train[data][i][j];
					}
				}
				float[][][] output = fcl.propagation(input);
				fcl.backPropagation(output, label[data]);
			}
			System.out.println("TRAINING EPOCH :" + epoch + " / 1000");
		}
		
		System.out.println("---TEST---");
		for(int data = 0 ; data < 4 ; data++){
			System.out.println("DATA:"+data);
			float[][][] input = new float[1][3][3];
			for(int i = 0 ; i < 3 ; i++){
				for(int j = 0 ; j < 3 ; j++){
					input[0][i][j] = test[data][i][j];
				}
			}
			float[][][] output = fcl.propagation(input);
			System.out.println("CLASS1:"+output[0][0][0]+" CLASS2:"+output[1][0][0]);
		}
	}
}