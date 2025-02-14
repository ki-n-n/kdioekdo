import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import time


class people_flow():
    '''
    Social Force Model (SFM)を用いて、指定した人数の人々が入口から入って出口から出るまでをシミュレーションする。
    
    '''

    def __init__(self, people_num, v_arg, cons_h, cons_w,cons_x, target, R,
                 min_target, p_arg, wall_x, wall_y, in_target_d, dt, disrupt_point=None, save_format=None, save_params=None):

        try:
            self.people_num = people_num

            self.v_arg = np.asarray(v_arg)
            if len(v_arg) != 2:
                raise TypeError("The length of v_arg is mistaken.")

            self.repul_h = np.asarray(cons_h)
            if len(cons_h) != 2:
                raise TypeError("The length of repul_h is mistaken.")

            self.repul_m = np.asarray(cons_w)
            if len(cons_w) != 2:
                raise TypeError("The length of repul_m is mistaken.")
            
            self.repul_n = np.asarray(cons_x)
            if len(cons_x) != 2:
                raise TypeError("the length of repul_n is mistaken.")

            self.target = np.asarray(target)

            self.acceleration = np.zeros(self.people_num, dtype=bool)
            self.angle = np.zeros(self.people_num)

            self.R = R
            self.min_p = min_target

            self.p_arg = p_arg
            if (len(self.target) - 1) % len(p_arg) != 0:
                raise TypeError("The length of p_arg is mistaken.")

            self.wall_x = wall_x
            self.wall_y = wall_y
            self.in_target_d = in_target_d

            if save_format != None:
                self.save_format = save_format
                if save_format != "heat_map":
                    raise ValueError("The save format you designated is not allowed.")

            if save_params != None:
                self.save_params = save_params
                if save_format == "heat_map":
                    if len(save_params) != 2:
                        raise TypeError("The length of save_params is mistaken.")
                    if wall_x % save_params[0][0] != 0 or wall_y % save_params[0][1] != 0:
                        raise ValueError("The shape of the heat map is mistaken.")
                    if save_params[1] < dt:
                        raise ValueError("The interval of saving results is shorter than that of updating results.")

            self.dt = dt
            self.disrupt_point = disrupt_point

        except TypeError as error:
            print(error)
        except ValueError as error:
            print(error)

    def __sincos(self, x1, x2, on_paint=True):

        # x1,x2の組の個数が2以上かそうでないかで場合分けする
        if x1.ndim == 2:
            if x2.ndim == 2:
                # 0除算を防ぐためにシミュレーションされていない人に対してははrに1を足す
                r = np.sqrt((x2[:, 0] - x1[:, 0]) ** 2 + (x2[:, 1] - x1[:, 1]) ** 2) + np.logical_not(on_paint)
                sin = on_paint * (x2[:, 1] - x1[:, 1]) / r
                cos = on_paint * (x2[:, 0] - x1[:, 0]) / r
                return sin, cos
            else:
                r = np.sqrt((x2[0] - x1[:, 0]) ** 2 + (x2[1] - x1[:, 1]) ** 2) + np.logical_not(on_paint)
                sin = on_paint * (x2[1] - x1[:, 1]) / r
                cos = on_paint * (x2[0] - x1[:, 0]) / r
                return sin, cos

        else:
            if x2.ndim == 2:
                # 0除算を防ぐためにシミュレーションされていない人に対してははrに1を足す
                r = np.sqrt((x2[:, 0] - x1[0]) ** 2 + (x2[:, 1] - x1[1]) ** 2) + np.logical_not(on_paint)
                sin = on_paint * (x2[:, 1] - x1[1]) / r
                cos = on_paint * (x2[:, 0] - x1[0]) / r
                return sin, cos
            else:
                r = np.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2) + np.logical_not(on_paint)
                sin = on_paint * (x2[1] - x1[1]) / r
                cos = on_paint * (x2[0] - x1[0]) / r
                return sin, cos


    def __force(self, x, v, v_opt, target_num, in_target, on_paint):
        '''
        人に働く力を計算
        '''

        sin, cos = self.__sincos(x, self.target[target_num], on_paint)
        fx = on_paint * (v_opt * cos - v[:, 0])
        fy = on_paint * (v_opt * sin - v[:, 1])

        for i in range(len(x)):
            if not on_paint[i]:
                continue

            on_paint[i] = False
            sin, cos = self.__sincos(x[i], x, on_paint)
            on_paint[i] = True
            r = np.sqrt((x[:, 0] - x[i, 0]) ** 2 + (x[:, 1] - x[i, 1]) ** 2)
            f_repul_h = self.repul_h[0] * (math.e ** (-r / self.repul_h[1]))
            fx_repul_h = np.array(f_repul_h * cos)
            fy_repul_h = np.array(f_repul_h * sin)

            # 人から遠ざかる方向に反発力が働くのでマイナス
            fx[i] -= np.sum(fx_repul_h)
            fy[i] -= np.sum(fy_repul_h)

        # 人と壁の間に働く力を計算
        fx += self.repul_m[0] * (math.e ** (-(x[:, 0] - 135) / self.repul_m[1]))
        fx -= self.repul_m[0] * (math.e ** (-(165 - x[:, 0]) / self.repul_m[1]))

        for i in range(len(x)):
            if not on_paint[i]:
                continue  # 既に消えているならスキップ
        
            left_clear = True  # 左側の扇形が空いているか
            right_clear = True  # 右側の扇形が空いているか
        
            for j in range(len(x)):  # 他のすべての人をチェック
                if i == j or not on_paint[j]:  # 自分自身 or 既に消えた人は無視
                    continue
            
                in_sector, side = self.is_in_sector(x[i,0], x[i,1], x[j,0], x[j,1], self.angle[i], 20, math.radians(90))
            
                if in_sector:
                    if side == "left":
                        left_clear = False
                    elif side == "right":
                        right_clear = False
        
            # 条件に応じて加速や移動の方向を変える
            if not left_clear and right_clear and i >3:
                
                self.acceleration[i] = True
                fy = np.array(fy, dtype=float)

                safe_distance = 0
                min_distance = 0

                for i in range(len(fy)):
                    front_people = x[:, 1] > x[i, 1]
                    if np.any(front_people):
                        min_distance = float(np.min(x[front_people, 1]) - x[i, 1])
                    else:
                        min_distance = float('inf')
        
                if min_distance >= safe_distance:
                    tau = 0.3 #速度調整にかかる時間
                    lambda_n = 1 #

                    if min_distance != float('inf'):
                        # 空間が開いているほど加速する関数（tanh を使用）
                        acceleration_factor = np.tanh(float(min_distance) / lambda_n)  

                        # 目標速度 v_opt に向かう加速度
                        desired_force = ((v_opt - float(v[i, 1])) / tau) * acceleration_factor

                        # y方向の力を加算
                        fy[i] += np.sum(desired_force)
            if left_clear and not right_clear and i >3:
                
                self.acceleration[i] = True
                fy = np.array(fy, dtype=float)

                safe_distance = 0
                min_distance = 0

                for i in range(len(fy)):
                    front_people = x[:, 1] > x[i, 1]
                    if np.any(front_people):
                        min_distance = float(np.min(x[front_people, 1]) - x[i, 1])
                    else:
                        min_distance = float('inf')
        
                if min_distance >= safe_distance:
                    tau = 0.3 #速度調整にかかる時間
                    lambda_n = 1 #

                    if min_distance != float('inf'):
                        # 空間が開いているほど加速する関数（tanh を使用）
                        acceleration_factor = np.tanh(float(min_distance) / lambda_n)  

                        # 目標速度 v_opt に向かう加速度
                        desired_force = ((v_opt - float(v[i, 1])) / tau) * acceleration_factor

                        # y方向の力を加算
                        fy[i] += np.sum(desired_force) 
            elif left_clear and right_clear and i > 3:
                self.acceleration[i] = True
                fy = np.array(fy, dtype=float)

                safe_distance = 0
                min_distance = 0

                for i in range(len(fy)):
                    front_people = x[:, 1] > x[i, 1]
                    if np.any(front_people):
                        min_distance = float(np.min(x[front_people, 1]) - x[i, 1])
                    else:
                        min_distance = float('inf')
        
                if min_distance >= safe_distance:
                    tau = 0.3 #速度調整にかかる時間
                    lambda_n = 1 #

                    if min_distance != float('inf'):
                        # 空間が開いているほど加速する関数（tanh を使用）
                        acceleration_factor = np.tanh(float(min_distance) / lambda_n)  

                        # 目標速度 v_opt に向かう加速度
                        desired_force = ((v_opt - float(v[i, 1])) / tau) * acceleration_factor

                        # y方向の力を加算
                        fy[i] += np.sum(desired_force)
                           
            else:
                self.acceleration[i] = False
                

        # 上記の式を定数を変更しながらシミュレーションの結果を考察

        return fx, fy

    def is_in_sector(self, self_x, self_y, other_x, other_y, self_angle, radius, theta):
    
    
        dx = other_x - self_x
        dy = other_y - self_y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > radius:
            return False, None  # 半径外なら対象外

        angle_to_other = math.atan2(dy, dx)  # 他人の角度
        angle_diff = (angle_to_other - self_angle + math.pi) % (2 * math.pi) - math.pi  # -π ~ π に正規化

        if abs(angle_diff) <= theta:  # 扇形内にいる
            if angle_diff > 0:
                return True, "right"
            else:
                return True, "left"

        return False, None


    def __calculate(self, x, v, v_opt, p, target_num,
                    target, in_target, stay_target, on_paint):
        '''
        人の状態を更新
        '''

        # シミュレーションする場所にいない人は更新しない
        x[:, 0] += on_paint * v[:, 0] * self.dt
        x[:, 1] += on_paint * v[:, 1] * self.dt

        # 誤って壁の位置より飛び出ていたら壁の内側に戻す 
        '''
        このコードを変更済
        '''
        for i in range(len(x)):
            if x[i, 0] > self.wall_x * 0.55:
                x[i, 0] = self.wall_x * 0.55
            if x[i, 0] < self.wall_x * 0.45:
                x[i, 0] = self.wall_x * 0.45
            if x[i, 1] > self.wall_y:
                x[i, 1] = self.wall_y
            if x[i, 1] < 0:
                x[i, 1] = 0

        fx, fy = self.__force(x, v, v_opt, target_num, in_target, on_paint)

        v[:, 0] += fx * self.dt
        v[:, 1] += fy * self.dt

        '''下記のコードにて、目的地計算式でx座標を考慮せず、y座標のみを用いた距離計算でok-修正済'''
        # 目的地と人の間の距離を計算
        target_d = (self.target[target_num, 1] - x[:, 1])
        for i in range(len(x)):
            if not on_paint[i]:
                continue
            # 目的地が出口であればそのまま続ける
            if target_num[i] == len(self.target) - 1:
                continue

            # 目的地との距離が"in_target_d"より近ければ到着したと見なす
            if target_d[i] < self.in_target_d:
                in_target[i] = True

            if in_target[i]:
                stay_target[i] += self.dt
                # 滞在時間の期待値が過ぎれば次の目的地に進む
                if stay_target[i] > (1 / p[i, target_num[i]]):
                    target_num[i] += 1
                    in_target[i] = False
                    stay_target[i] = 0.0
                # 滞在時間が過ぎる前に遠ざかったら、また近づくようにする
                if target_d[i] > self.in_target_d:
                    in_target[i] = False

        return x, v, target_num, in_target, stay_target

    def __initialize(self):
        '''
        シミュレーションに使う変数を初期化する
        '''
        x = list()
        v_opt = list()
        v = list()
        p = list()
        target_num = list()
        in_target = list()
        stay_target = list()
        on_paint = list()
        for i in range(self.people_num):
            # 入口を広めにして、スタート位置はランダムにする
            x.append([(self.wall_x * 0.1) * np.random.rand() + self.wall_x * 0.45, self.wall_y])
            v_opt.append(abs(np.random.normal(loc=self.v_arg[0], scale=self.v_arg[1])))
            v.append([0, -v_opt[i]])

            p_target = list()
            for k in range(len(self.p_arg)):
                for m in range((len(self.target) - 1) // len(self.p_arg)):
                    p_candidate = np.random.normal(loc=self.p_arg[k][0], scale=self.p_arg[k][1])
                    # 次の目的地に移動する確率が"min_p"より小さかったり1より大きかったりしたらその値にする
                    if p_candidate < self.min_p:
                        p_candidate = self.min_p
                    elif p_candidate > 1:
                        p_candidate = 1
                    p_target.append(p_candidate)
            p_target.append(self.min_p)
            p.append(p_target)

            target_num.append(0)
            in_target.append(False)
            stay_target.append(0.0)
            on_paint.append(False)

        # numpy.ndarrayにする
        x = np.asarray(x)
        v_opt = np.asarray(v_opt)
        v = np.asarray(v)
        p = np.asarray(p)
        target_num = np.asarray(target_num, dtype=int)
        in_target = np.asarray(in_target)
        stay_target = np.asarray(stay_target)
        on_paint = np.asarray(on_paint, dtype=bool)
        return x, v_opt, v, p, target_num, in_target, stay_target, on_paint

    def __start_paint(self, x, on_paint):
        '''
        入口が混んでいなければ入場を許可
        '''
        tenth_started = False
        for i in range(len(x)):
            if x[i, 1] == self.wall_y and on_paint[i] == False:
                for k in range(len(x)):
                    if on_paint[k] == True:
                        if np.abs(x[i, 0] - x[k, 0]) < self.R * 0.2 or np.abs(x[i, 1] - x[k, 1]) < self.R * 0.3: 
                            break
                        ''' 0.5 -> 1.5   0.5-> 2.0'''
                    if k == len(x) - 1:
                        on_paint[i] = True
        #10人目が入場したタイミングを知らせる
        tenth_started = any(on_paint[i] for i in range(9, 49))
          
        return on_paint, tenth_started

    def __judge_end(self, x, target_num, on_paint,counts):
        '''
        出口に着いたら描画や計算をやめる
        '''
        target_d = np.sqrt((self.target[-1, 0] - x[:, 0]) ** 2
                           + (self.target[-1, 1] - x[:, 1]) ** 2)
        for i in range(len(x)):
            if target_d[i] < self.in_target_d and target_num[i] == len(self.target) - 1:
                on_paint[i] = False

        end_flag = False
        fiftieth_exited = False
        
        # 50人目の計測が終了したタイミングを知らせる
        if len(on_paint) >= 49 and on_paint[49] == False:
            fiftieth_exited = True


        if np.sum(on_paint) ==0 :
            end_flag = True    

        return on_paint, end_flag, fiftieth_exited

    def __paint(self, x, target, on_paint):
        '''
        エリア内にいる人を描写

        '''
        ax = plt.axes()
        plt.xlim(0, self.wall_x)
        plt.ylim(0, self.wall_y)
        for i in range(len(x)):
            if not on_paint[i]:
                continue
            # 人の描画
            if self.acceleration[i] == True:
                particle = pt.Circle(xy=(x[i, 0], x[i, 1]), radius=self.R, fc='red', ec='red')
            else:
                particle = pt.Circle(xy=(x[i, 0], x[i, 1]), radius=self.R, fc='k', ec='k')
            ax.add_patch(particle)
        for i in range(len(target)):
            if i < len(target) - 1:
                # 目的地の描画
                obj = pt.Rectangle(xy=(target[i, 0] - self.R, target[i, 1] - self.R), width=self.R * 2,
                                   height=self.R * 2, fc='y', ec='y', fill=True)
                ax.add_patch(obj)
            else:
                # 出口の描画
                exit = pt.Rectangle(xy=(self.wall_x * 0.45, 0), width=self.wall_x * 0.1,
                                height=self.wall_y * 0.01, fc='r', ec='r', fill=True)
                ax.add_patch(exit)
        # 入口の描画
        entrance = pt.Rectangle(xy=(self.wall_x * 0.45, self.wall_y * 0.99), width=self.wall_x * 0.1,
                                height=self.wall_y * 0.01, fc='r', ec='r', fill=True)

        ax.add_patch(entrance)

        left_wall = pt.Rectangle(xy=(self.wall_x * 0.43, 0),width=self.wall_x * 0.01, height=self.wall_y,
                                 fc='r', ec='r', fill=True)
        
        ax.add_patch(left_wall)

        right_wall = pt.Rectangle(xy=(self.wall_x * 0.56, 0),width=self.wall_x * 0.01, height=self.wall_y,
                                  fc='r', ec='r', fill=True)
        
        ax.add_patch(right_wall)
        ax.spines["top"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["left"].set_linewidth(1)
        ax.spines["right"].set_linewidth(1)
        ax.spines["top"].set_color("coral")
        ax.spines["bottom"].set_color("coral")
        ax.spines["left"].set_color("coral")
        ax.spines["right"].set_color("coral")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.plot()
        plt.pause(interval=0.01)
        plt.gca().clear()

    def __heat_map(self, x, on_paint):
        '''
        ヒートマップを作成
        '''
        map = np.zeros(shape=(self.save_params[0][0], self.save_params[0][1]))
        # 指定した行、列1つあたりのx,yの範囲を計算する
        rough_x = self.wall_x / self.save_params[0][0]
        rough_y = self.wall_y / self.save_params[0][1]
        # 各人がヒートマップ上のどこにいるか計算する
        location_x = (x[:, 0] // rough_x).astype(int)
        location_y = (x[:, 1] // rough_y).astype(int)
        for i in range(len(x)):
            if on_paint[i]:
                # 計算した場所が保存するヒートマップの範囲外であったら範囲内にする(一番端にいると範囲外になる)
                if location_x[i] >= self.save_params[0][0]:
                    location_x[i] = self.save_params[0][0] - 1
                if location_x[i] < 0:
                    location_x[i] = 0
                if location_y[i] >= self.save_params[0][1]:
                    location_y[i] = self.save_params[0][1] - 1
                if location_y[i] < 0:
                    location_y[i] = 0
                map[location_x[i], location_y[i]] += 1

        return map

    def simulate(self):
        '''
        人流をシミュレーション
        '''
        # シミュレーションにかかった時間を記録
        start = time.perf_counter()
        x, v_opt, v, p, target_num, in_target, stay_target, on_paint = self.__initialize()
        end_flag = False
        steps = 0
        counts = 0
        total_time = 0
        calucurate_time = 0
        tenth_started = False
        fiftieth_exited = False

        if self.save_format == "heat_map":
            self.maps = list()
            save_times = 0
            passed_time = 0

        while not end_flag:

            x, v, target_num, in_target, stay_target = \
                self.__calculate(x, v, v_opt, p, target_num,
                                 self.target, in_target, stay_target, on_paint)

            on_paint, tenth_started = self.__start_paint(x, on_paint)
            on_paint, end_flag, fiftieth_exited = self.__judge_end(x, target_num, on_paint,counts)
            if self.save_format == "heat_map":
                if passed_time > save_times * self.save_params[1]:
                    self.maps.append(self.__heat_map(x, on_paint))
                    save_times += 1
                passed_time += self.dt

            self.__paint(x, self.target, on_paint)

            steps += 1

            #10人目が入場してから50人目の計測が終わるまでの時間を計算する
            if tenth_started and not fiftieth_exited:
                counts += 1
                
            
        # シミュレーションにかかった時間を計算

        calucurate_time = counts * self.dt
        total_time = steps * self.dt
        # シミュレーションにかかった時間を記録
        end = time.perf_counter()
        
        # かかった時間を出力
        
        print( " 総合時間 " + str(total_time) + "s" )
        print( " 測定時間 " + str(calucurate_time) + str(counts)+ "s")
        return self.maps if self.save_format == "heat_map" else None
