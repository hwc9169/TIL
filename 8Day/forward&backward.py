#모든 계층은 forward()와 backward()라는 공통의 메서드를 갖도록 구현한다.
#forward() 순전파 backward()는 역전파를 처리한다

#곱셉 계층

class MulLayer():
    def __init__(self):
        self.x = 0
        self.y = 0

    def forward(self,x,y):
        self.x= x
        self.y =y
        out = x*y

        return out

    def backward(self,dout):
        dx = dout*self.y
        dy = dout*self.x

        return dx, dy

class AddLayer():
    def __init__(self):
        self.x =0
        self.y =0
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x+y

        return out
    def backward(self,dout):
        dx = dout
        dy = dout

        return dx,dy

apple =100
apple_num =2
orange = 150
orange_num =3
tax = 1.1

#계층들
apple_mul_layer = MulLayer()
orange_mul_layer = MulLayer()
add_layer = AddLayer()
mul_layer = MulLayer()

#순전파
apple_price = apple_mul_layer.forward(apple,apple_num)
orange_price = orange_mul_layer.forward(orange,orange_num)
sum_price = add_layer.forward(apple_price,orange_price)
price = mul_layer.forward(sum_price,tax)

#역전파
dprice =1
dsum_price,dtax = mul_layer.backward(dprice)
dapple_price,dorange_price = add_layer.backward(dsum_price)
dapple, dapple_num = apple_mul_layer.backward(dapple_price)
dorange, dorange_num = orange_mul_layer.backward(dorange_price)

print(price)
print(dapple_num,dapple,dorange,dorange_num,dtax)

