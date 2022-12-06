import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Annotate(object):
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        #self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.moving)
        self.flag = 1

    def moving(self, event):
        print(event.xdata, event.ydata)
        if self.flag == 0:
            self.rect.set_width(event.xdata - self.x0)
            self.rect.set_height(event.ydata - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            self.ax.figure.canvas.draw()
        else:
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
            #self.ax.figure.canvas.draw()
            #pass
            # self.ax.figure.canvas.draw()
        # self.ax.figure.canvas.pause()


       
    def on_press(self, event):
        print ('press')
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.flag = (self.flag +1) %2
        if self.flag == 1:
            self.x1 = event.xdata
            self.y1 = event.ydata
            print(self.x1, self.y1)
        print(self.flag)
        

            

    # def on_release(self, event):
    #     print ('release')
        
    #     self.x1 = event.xdata
    #     self.y1 = event.ydata
    #     self.rect.set_width(self.x1 - self.x0)
    #     self.rect.set_height(self.y1 - self.y0)
    #     self.rect.set_xy((self.x0, self.y0))
    #     self.ax.figure.canvas.draw()
    
a = Annotate()
plt.show()