from time import sleep
from BMI160_i2c import Driver

sensor = Driver(0x69)  # use 0x69 if your board is wired that way

while True:
    gx, gy, gz, ax, ay, az = sensor.getMotion6()
    print(f"gx={gx} gy={gy} gz={gz} ax={ax} ay={ay} az={az}")
    sleep(0.01)

