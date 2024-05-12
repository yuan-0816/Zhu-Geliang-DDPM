"""
Yuan @ 2024.05.12
This is a script to print the artwork and information of Zhu Geliang.
Zhu Geliang is alive!
"""

def PrintInfo():
    print('------------------------------------------------------------------------')
    print('This program is made by Yuan')
    print('It is a generative model implemented using DDIM to gernerate Zhu Geliang')
    print('Zhu Geliang is a living being!')
    print('------------------------------------------------------------------------')

def PrintZhuGeliang():
    artwork = """
     #######  ###                          ####             ###       ##                                           ##
     #   ##    ##                         ##  ##             ##                                                   ####
        ##     ##      ##  ##            ##        ####      ##      ###      ####    #####     ### ##            ####
       ##      #####   ##  ##            ##       ##  ##     ##       ##         ##   ##  ##   ##  ##              ##
      ##       ##  ##  ##  ##            ##  ###  ######     ##       ##      #####   ##  ##   ##  ##              ##
     ##    #   ##  ##  ##  ##             ##  ##  ##         ##       ##     ##  ##   ##  ##    #####
     #######  ###  ##   ######             #####   #####    ####     ####     #####   ##  ##       ##              ##
                                                                                               #####
    """
    print(artwork)

def PrintEverything():
    PrintZhuGeliang()
    PrintInfo()


if __name__ == '__main__':
    PrintEverything()
