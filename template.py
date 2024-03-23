css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 10%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCBUVFBcVFRUXFxcaGxoaGhcbGhodGhcXGxcYGBoaGhsdICwkGx0qHhoXJTYlKS4wMzMzGiQ5PjkyPSwyMzABCwsLEA4QHRISHTIpICoyMjIyMjQyMjMyMjI0MjIwNTIyMjIyMjIyMjIwMzQyMjIyMjIyMjIyMjIyMjIyMjIyMv/AABEIAPsAyQMBIgACEQEDEQH/xAAcAAAABwEBAAAAAAAAAAAAAAAAAQIDBAUGBwj/xABDEAACAAUCAwYDBQcCBAYDAAABAgADERIhBDEFQVEGEyJhcYEykaEHQlKxwSNicoLR4fAUMyRTsvE0Q5Ois8Jzg5L/xAAYAQEBAQEBAAAAAAAAAAAAAAAAAQIDBP/EACQRAQEBAQACAgEEAwEAAAAAAAABEQIhMRJBAyJRYZEygfBx/9oADAMBAAIRAxEAPwDraqajEOzGBFBmAZgIoOcIRCpqdoAShQ5x6wc7NKZ9IOYbsCBLNu+IA5RoKHENupJJAg3UsajaFq4AodxAKLim4hmWpBBIpB92a15bwtnBFBvACaajGfSEycVrj1gkW01MHM8W3KAKaKnGccoWjAAAmkJluAKE/r+UNzDUkiBoyprsd4edgQaEQQmeRhlMEYhiaXKFDU4hU3NKZ9IKa4IgpTAbmClSTQZx6wmYpJqMwbi7IhSMFFDvAGrAAZG0MqpqMGFNLJyOcLMwEUHOAExgRQZhEoUOcesBVKmp2hUw3YEAU7NKZ9IVKNBQ4hMs274rBOpY1G0ATqSTQQ5WArgCh3EJpAEJRGemYUXuwILva4pviDstzvAEBbkwGF22KQLrsbQCbPOsAavbgwRl1z1gBLs7QO8txTaAMzRt7QQl256Qfdc6+cY7tH23WWDLkATJhJUMMgsN1QffI5moVeZwRATONdsJMmaZLXVUBnahIQHatNt4udNNSYgdGDKwqGBqCDzHKOManXzXZ0mqGLN3hmOSEUlaVBwZxpQFVoo2rGw+y/iKOs7TpMMwS2VwbVVQHuqEVdlqlefxbmJOvOHXNzfpvFUADy54H5YhQccs+kJEo+W/mfz2gOQuST05R0cy7/b1IhQaClyFXYU9z9c5h0LEJptmoKmvyP6QTOo3IELAIHX1pU/LEAg4x652/rBSUHPb02gmUk1hVvT/AD2huTNqMi09D06jqIGnROAxQ4hIlEZ6Zg2AMGWOx2OKxMalAvdgQQFuTB2W53grrsbRFBhdtygw9uDBE2edYAS7O0ARl3Z6wu6EmZbim0HbAGZYGRyhCsWNDtCVckgVh2YoAqMGAJxbkQSC7flBSjU5zBzsUpiAJ2KmghaywRU84EtQRU5jL9tOODTy7bqX1DU3CfDvmlWIUY5/MKHtp2sZj/ppBLXEr4TQzSPiFw+CUv3m3PLFLud8VlshZXraFF7AWmaKAiWn/L061AJ5moyamLc8SkrNaU0tDchZ3tFolmjIkxTUgGpagNBVSVOaVesNrF1eiFaIpobZeSQa1uWuVrimTgUOLtvh0lknmIkqXNnKkvCqKGW1poqnwmwAVYbDrXptGp7AzpXD9Q5mubZiqhJtFHB8PguL0NWyQIyOq4/MVCqsQGFK/ecY3Y8sf2jScK7L6zue87qYZzqSoIYd2h5C7HeEdaWjG5MdOY49V2zTamXMW+W6uDzBr/2h1zg+kcI0fZvi2nUvJScr/FYCCMnZvF8W5xjIycgX2j7TcalgCfoJ8wUFSkti3tbn6xpn36dcrArFB2b43M1KkzNPOkkbiYjpnyvAr7YiybVhSQxJz028vOGGp1YOsUGs42ktqNNRK/CpIqR1ocmJcjiYKg039vodomHyWDGppz5+nSHCIgjXr+GHRrV84YvyLm45QlZvl6iGZ2pBwPnCQ8MTUvenNT8/SDcW5EMyJnI7H84dl1qQ2af5WJW5QQXb8oJ2Kmgg52KUxCpagipzEUFlgip5wV0IdyCQDDlIBTkUO0NSga5glQg1I2hx2BFBvAFOyMfSBKxWv1gpYtycQJgu2zAFNFTj6RxjtjxEanWWs37Jbnmcx3Eq4ewYrNBXPisNKx1rjOr7rTTGzcFNKb1bwgj3IjgGum3LqHr/ALkxNMrLvYpBmMOQVllyj/M3Ktc1vn90ZGeY1zGjz3Dv+6reJf5VQFh6EQ12k1tWouK5I6CtFX2tp52LGp7Ndmu8lvOms0tmZxLwCFSoFxFRUUFoAIwW/FDJ7ImTO72eVmygQEsJBdzW0MCPDQAdQaVB3peZrPfUk2/Sf9m3ZZRTVTlBcEhFOQuxup+IGvueojpZmjeKXSzVRFVQFAGw2HM09yYfGqjrI41YcC7QS9RLWbLJeWxIBIKnwmhwfMRcrrEPOnrGZXUQ3rQZkt5YdpZZSt6GjLXmp6xcNa06pPxCM92t4vK0kltQ1WAIFqkVLMQFArtv9IY0psRULM9oAuY1ZqDcnrFH9oWqt0LtarAPLqrKGUgsAQQfInOCNwQcxMw9rjgmt/1EiXPZApdbgta2qSaCvM8/eLC+Kfswy/6PT2ii92hUVJoCKgVOTiLItBDweFrMiLdC1MaQ+wuKmpFpqPW1l+VGMPhj1hiT+p/Mw+FjIUsw03yOkWLNVVYc8H/PX84qnWmab48/L9fnFlpW/Z/wn8iDE6b4qRKxWv1hM3fH0g5gu2zCkYKKHBjDoUhFBWkN0hLISagYhy6ABmg4znEIVCuTB91TNdswL7sbQBs12B9YIG3fn0gW2535QKX+VIDM9t5/7EAEqTVwcU8A2NSBSpB3GF9I5LrtBKlyZKBzeiu6qzIoYOApdgd8KAKVApkxv/tfZl0yWk1zkYx3kkn6A/OOba+WGOmvrc0lFZjSpDy5TLUmv3mb5mMdN8+I6Pw+WFkS5fIIB68z9axU9rZ5TRzGJyGGfJpgWlf4WIPkTF3NOcbcvSM726/8FNxXMvHX9qkb59ufU8K7T9oJniBozgVVcAuuMitBXfy26w/pu18u6yZdLalaMCCB59PcxiuNtbMUfeANDsVqWApT+EQzM45VSrLXkDio9Dzi9bOk5zrmb7dc0nEUmAFHBr5xOR45JpZX+nQuSUYi7c0FeoP3vXn6Zn8O7eulO8S5eRB8VOpB/r7Rudfuzef2dSR4y/2kza6MSwCWmTEAUAljbVjRRk7D5xE4R2lkSpKhEmKlTbe2SSamhdqtvDXFOOuXZZiGUhl/s5ylWZanxFGBKHYVBocbrub1fDPP+Wa2XZqWV0enUggrKQEHcEIAQYsqRXdmDdpZVWuIW0tnJUlSfFnlzzFt3cIX2jpLFBQDbpDyJ/lTBouIOKhekUAEXE5NampHl5RZSZRYEjlFOiEXMpyTsdsHnT3+cW2m4giJRjQxmrCmlQ5oFNjKa/CNxQmqnccj5RXajj0sE/pE7g+sE1WdcLsD1I3iX01z7TZbWjPPOINkuyPrDWk8aL5Y+WP0h2+3G8YdBiYBg1xBWQO6uzXeBfAEJpOMZxCigXIhRQAEgQ2jEmhyIA1N2D9IDG3bn1g5vhGMQJXi3zAYj7UJd2luI2Sb9FV//oY5hqgTO0YpW6RpB6E3An2AB9o6/wBv9LfpmUcw6D1my3lj/wBzLHMOHBZkzQzNx3bt7SVVFHu10c+v8nXmfpbUfqf7fSkUvbD/AMFNNK2mW1OoWbLMWjMFFWalK1NaDOSYzXa7jksad5aks0xQAQDbS5DWvPlt1Eb5vlz7nhl+1/DGlvLm3K0t1BDKa08ZoG6EqQw6hvIxWdn9GrzFJF1CWzW2i7VA3q1BTzibo+LPKQGtwBbBqKgB29sjbIFdo0fZ3UafUk0Xu5hdEYqKAhiSoIGGBUPRgAQVzUERufqt1z6vw5kk07qeAS5ssajWzG08kkm1aGbOOQLAcKtADUg16DcxW4dwaaQijVSTsJhZWB/iUjb0IjVanhK63VnvCe7RAqLyUEVwPl8oznaTswdNV1ZSnSufkcxcTfpF7SdlpkonUtbO04C928upAXGHFPB4qnNRU0rFWJrTdPNUYKMGUVa0AG0t6EXH+UdI3PZLiTJLVWNQwyDkEHGx6ikU3E+DpJ1UxZeJOolsQK/A+xUfu+IkdNuQjXPljqZPH0uvss1V2leXWplzGHs3j/MsPaN8gjlf2dnuZ2plclsF34mBcFq86x0aVq4T0t9pCoKY2qfLZiDEae9oP+Z5RLkzwR8/zMN6lAQPMj894qKuZq7RTpFdO1zsaAfOLfUcOuiq1vZ1WGZhXztVqezgj5CsAhBeaGcl34V8RHqFzG34fqpQokuYjBUqqqQXIXBY+/L/ALDm6dn54aYwmTTLWUyKL3IM13IExLTaAJecUKsu2axpuy3DjLcMxLOakuTVixGSSdyYz1Nb5uNbp0MtFXnSp9Tk09zD6pdkwJXi3zCZjEGgwI5ugNMIwOULtg1UEVIzCLjAIWtRvD0ylMfSAzgggGG5akGpwIASt8/WDncqfSDm+IYzAleHfEBA4zIL6aaB8YUstfxJ4l+ojkGiJVpNf/LnTJYqCP2dzhKUFBh13pgeUdp1Eu4EZIIINPMUMcm7W6JpImgD/bJcG0fCHlFj1/2n+ch/IRz7nmV1/HblkQu1wdZ8uSSCk1HC12LtLV5JHneAvuYhTeFidpAVBLLLuUDnYBLI9hJU/wD7BGi4+om6bSaoAEypiqxOyjcE/wAKu59UEVOg1HdTDKVrWDGbJu2daBXlnyKLKcehxvHb8dmeXl/Nzb14/wCxgFWssGtRjGa1NymL7gEwaeUkwigM1Zh/gTwBq9CTMpFtxLhmmDXiTNAYlu7UFpdzUJtdR4lr927GRdEXi8xQnjFKlTbWlLfgQAeYBPIUAh8c2r8/lkxtdPqGV3s+NpbW/wASmv0Vl+RjK8U0MxazNRMZychak1/QRK4FxAzfArhZstlox2vp4SR+A1ZDXbJ5RM41pxqfCaSJ64aVMJCk9UehBHrF5uw7mdKvs3Imah5vioJaqQBsKkgAeVFPyiRxq4Mld1DUJ25f0im0+tncP1IWYtpYAFagiYhNAVIqGHQjnUdRGi7WU3yKq3tiv9Y1zNZ6tk1XdlJbJMnvWqv3ZVgcMpvav1+lecaddXSMh2IlsulqchnYrvQKKCg/mDH3i7nOwBIGff8ApE53PLXWW+F/o9fUChxQflFpK1VaesZGTOpTf5GLHT6unX5GNM41aTQd/wDPfaFMisDQg9aHaKaRrx5/I/0iWmtU/wDY/wBIIncH4UA8xwSK2j2uuYU86DMSJCAeUNcH1o72wA0dSa5oCtKD5E/KBMm2uy9CRGPtv6W3eVAp70h6Vtn6xX6OabhSJsxSTUCM2Y3zdE9amlYcg1YAAE5hFIiiEojPTMKZgwoN4He1xTfEFZbneACi3JgOLtuUAtdjaB8PnWANWtwYynbPQK1HYeB1ZHxU0COf/jbUqPOYsaqy7O0QONyy0h1VbnUB0HIuhDoD5EqAfImM9zZjfHXxsrmXYcmdpZ2imUL0dKVr+0lk49wGr/EIzHaKVfLEwVvlsA1Kg1AVWp5BHlfI+cPcJ4umn4g7SiWluyTEpSrAsFNan4mNrGNX2r4aVdnRVMmaHmE/xqtBX8JUzB5EocRfx2er9sfm5s8z3HOeE6yYzBGZjW8VJP3UdxU7n4YYXxuXmE/GqW/uqhJPlQG4eZhLI8iZU5tcMtcXpS0qcYqpIPrD0/Sd4b5TA1Vv2bG11ZsMaHGcj2xXeN3m3IxzeZtn2TwifbqO8VySRMZ5dCBQLWlTg1NDjpHU+GcWkamWqzqF1AAmHelMB/wmnXeOXcP0NkzJBmMWCSwa/FWt5Gwpy3P5S+IdnZ4fvEdrqCtDa1N+W35QzKb8p/DrFkpAPGhC5UkqbT1UnY+kYftVxFG7y03KiEMQaZbBFeRoenOMhqjOTwze+tNMsWIodiGuII9/1ix7M6WXMmdyrlhersLSCEQgsCdjUqq4/FGpXPrnY2fDNCJUmWgFLVFRWtGPibPPJOYkGXEyysAy40uqqalueX5efpCQ5EWM2VEKZKp5D8v7RlSk1JiQusMQhLhSyyTFgveD68q6t0I+XMfKsXendXmlnNF39Sf0jMaNCDFj2g0815feaY2sCCQcVBUXU5A3CohUa1+IyZY+IAeQp+cRNJ2ikvNEpWqzVwM7AknG20cc0vZPXaycO8Z6EmrsTYgWlaU3ORjnX5dX7Ldj5WjFyu0yYVtLt03IVRtsN6nzjFxuStE0snI5wu6E95bim0CyMtlGUBnOMwlXuwYSrkmhO8OOoUVG8ATLbkfWCAu35dIEs3YOYEzw7YgAzW4EGssHPWDRQwqcmEM5BoNhAcH4/wBlCmre1wn7R7ARmjOqrzphZkoDn4lFCSI6LwRhq9G8kkGbIJXaoIK3KCDujKSKHlET7QdGbyyrkoSp595ZMmA/+pp9MPeKvg/EW0+uUr/tzQZZU3Bad43dG+tBiwYGLjXy5b+rHaz9MrPa/hBNRK8SmoMljQj/APE5/wClj6E4EZPWad0qFU4Bww8S4FA4YVB+LfrHUu1eiCzS61CObxyKsDR0PmGrUecVaTCRmjEbXKrU9KjEduet8Vw64y7HOdPqnlsjqFLKQ1MWkguaUB2rSNXpO0Et6XK8slahlym1CB0z5GJmp0EmZUtJlgmtWVFU1O5qoEY/Waebp5glhmWW7UFNiDyPnXeLYzNXnFJRmqStSKjNKC3Cg1PlnHWLLsXoll97NIALuZabfAhyajqxp/JGS4UrtNWW0x3WtbQ5tBG9VOP6Rq+B6ktKVVILS6q4H4qlj9WMWUvv/TYI8PAxRafX8jgxPl6kGNaxZiU6xD1CAih51FPLnElZkAICbqZ5eQiohpIpgbYA/dFAKef94lydPC7YTImMAbqHJtz93lXEUSpcrMXempSkZz/WfhUs3JRSpPQRDSfxRgTL0ol12MyZLFB5AMfyjNWOh8PQWhNgK7eZrU/l7RNZrcCKDs1oZ0uWGnzLpjeJgKHPLxfeXegAUDpXMaFFDCpyY511noXdhs5zBXwTOQaDYQu0RFKZRQ4ENSiSc5hKIQRiHnYEUGYBM7Axj0gSs1rn1hMoUOcQc7NKZgEzTQ4x6Q8iigqBCZRoKHENupJJpiAyfbEHvJJ5YB99Zok/6HmfOOUdodUbdOtisWly6sR4gw0mlm1B5eJzHUu1nE5cx1Etu8MsEPYCaN3unmW12rRK7xzbiOkrLk3ywWCUZi58JXTSZZoAKEEIBzyCdiI5WX5a7zqfDFjp+0r6mXLlzR+1UUd6/G6qLTSnxGXaSdjaNsw1qtGsyy5mW1gwKmhqOvlGQ004q6zJZo9SlpyAEQHb8JoaDyjY6fULMrT4hS9fwmlfl5xv1XKXYkIMwzxTSK8skithDjrVDcaeoBHvDw6xI0iAVyWBJNGNaV+6Og8o6zzHK+KwXZ2Yyd7dsiNMJIyHZKD6t9IaHFe6ngyatLQBANu85u7ebOWYHkKCLTW8PmSZhllWKTC5EzcTCUe0E8iB90/WgMUE2kskyrhU0BahYY2FBTzPljrEsyQ5u2t/ptVLmrXnz/Ep6GHgrrlTcPrGPSa8tFLNW4mkwqQAcblcr/YxacL1etajCWjqeZYAU/iB/MGLo0UrVg4JIyPpyNeUWMvVDrFe8sN8S/XIMNPpmX4TUdG/rGp0z8V4s27pbihrv/b84RMUmKiVrSuGBH5RZafVqckim/tCJZiRo07s3c85hqfrJzue6YqFFzsalVWtBUDckkAAZJMT5Uu4ARf9m5Alll/Hz51ANPoTDr0s9rDgKTe6UTgtwGAAa0OfFkgN5AkDqYmzTQ4x6Qc3xUpnr5QuUaChxHJ1GiggVAhusE6EkkCHKwAMwEUHOEKpU1O0ASqZrtmFX3YgA5uwICG3eCC25OYBF22KQBMpY1G0Q+Lz7ZVvNiF9jk/QH5xODW43ij4y9zqvQXH1P9h9YDkPHpUyRrZ6EUExWmI/7rqqCh6h7Rjp5xW8fm3TJQDE2rNpknwgLLU+9n5x2fi3BZc/TIJoN1TY4+JK1IoemK0OPpTkHazgMzSuXYXSu7KJMA8LN3l9v7r0J8J6YrvE6n2vF+mTWZQimSST/DUAZ9Fp840TzXlzu8QfDS/ejO7JVD5qhHocxR8Pli8FwbV8VKfHQiiknqQoPrGt4ZIaxzMNan5sCWmN7m4eijpFk2yM9dfDm1baeaJiK6ZVhUH/ADnEiWsUPYDVXd7KOR/uJ6MxDAeVaH3Ma95FI3yz1TQFRQgEHcHYxhdR2fs1LV/2VHeIWOKsTUN1KlSSd6IK7xp+EmWJs9EmvMa+rI5qJZzhMbf0Hu32n0zvJIQXAOrOtaF5exUHrd3fy9jf5Zu+oxbTazDWqg0BlPUBkG1GXZvvXciTneLnhEsgzEkzAruA/iAdkEtXZ7bCVaqg1yK02BpSCgUeEzGUf8uav/SxFKecdC7E8ADuqstAoWY5AorqblSXubgTcx2GKGtaxy69/wAunN8Z9JPAOzOqnyw80rKBAK1BvYUHiZASEr0uMT53ZKcvwvLby8Sn8iPrG4DW4OecApdmNaZHMtZwibLH7SUwH4gLl9ytQIqX0nOWaeW4+UdkE2mKbYiFqODyXy8tG6m2jfMZi6mOXabiryvjBp1GRGi4B2gEyci4pXNTTlQAdWrSgGSYvn7K6RziWwr++36kxL4dwLT6Y3S5ShtrslvYnb2i/JPispZtrXFc0gmUsajaDIu2xSDDW43jDY1cAUO4hNpgGXdmu8HdAJE0nFN8Qqy3MGZYAqOUIVixodoAw12DiATbtmsG4tyICC7flAEEuycRnpS95MJ/E2P4eX/tEWvF9RZLIBpd4R77/SsR+EoAGmdBQfKp/T5xYlO8QbIQbD/B9PzhhdAk5JkuaoZGADK2x5/MYIPKAWqSTzi000u1QPcxplx3tB2BOidpss36YXObvilUBIVuq7gHzFdqmqkG3TuzGpKu1PagX2AAjvToCCCKg4IOxEck7e8EXTo7ShSUzBaf8su9W/lzj5dInHMl1j81vXM5/mf0w3ZzXLJ1CNS1baHxV8Ozb7Uur7R1R0BFRkHIPURxzTuEMpiMA0b+FsEfL841XC+OPoy0uaS0gPap+8isLl9VAMa5vjavWTrI1R06qzMFUM3xMAAWoKC48/eG5soOCh2dWTn95So28yD7Q9p5/eqXAWwnwMGuvWgycC01rjP9DTDKejA/IiNIyHCZBmTERHnLUiiMgau1BWlFr58hnpHY+zehMiSCykPMN7qxqUqAEQnnagVfYxgeBcEmSmlMGcK8/urWa4EKZpelDXwiVsRkvzpHVUN28cXWTAC3ZOIBe3EBzbgQpVDCp3gou6rmu+YSJpOKb4gNMIwOULMsAVHKAKy3MFW7BxBKxY0O0KcW5EARNu2awYW7O0BBdvyhLNaaDaABmW46Qq2DVARU7mCugG0ckjMOzFAFRgwpmFDkQzLBBzj1gDlGpzmDnYpTEHONRjPpCA4RWLYAznoIDMdrNC2qsld40tV8TsvxMT8K15DFfenOMZxZeJ8PRRIdp8kjxu4LANUkkqDfLUC0CpYdTXB3SMXYud2Nf7RNDNaADSmfX1i4zayHCuK651LHTpMUMUJRrTcoF1A24DErXnaY1XB+0KTCJcxWlTR9yYLS38JOG9okyZIYW4WnkKEnp71PWI/EdArLbMW4b1Aa5T1U5tPnURpldO4AyaRnuJadJquji5HqCPI/kedeVIYlatkIlzHLp9yYabdGIJBzivpWmIlTDgnf0iyJa4hx/gDyH7r4vHgilSh+BqedPniHe2WkmGZJlhSHZCaY3FBcc0IoD6RqftOk0GlnAf8AmBGrzDUda+YKk+RjNdrZpCqijLqBU5ZZZaZeL+hKJig5jnC+OfDMu9zffkfY7V93NEqW90s1v3pcdivKm+R0O2BG6s2jk/DZkxKzJeDLINQKkoDSnufEesdS4LrBOlLMHMDHmQDjyyIzzfGOnU861mikXTJM5mLBi5SVaoWXMtImOpUAsW8Xx1pcfbRzsUpiKPhkr/ZH4ZRf3mP/AEi7k4rXHrErUHJFRnMJmMQaDAgTcnGfSHJZAGcesRRqoIGIYRySMwHU1NAYfdhQ5G0ATqAKjENyjU5zBSwQc49YXONRjPpAFOxSmIVLUEVOYKVitcesImipxn0gA7kEgGHKQaMABUiEUgCWWQanlC3YMKDeAZoOM5xCVS3JgBLFuTiK7jM8G1BzyfSuPr+UT50wFSdguTXpFGtXYsdz9ByEWJTkmXEhVgIkOqsVg9pyADWEsenyP6dIILCoor9ToFatMV3HLpXyPLz2NYoOOz5+jlCZLlmcgZQyVJKSzWrqQCQAaVBxTOKRryIIw1McV7Vdpv8AVk6ZpRlMgvW43EzFVWQCmwKFvW6G+0umuWVNqFsHMkCyYoZDWn4lH/8AUar7SOy4ZDrZKEzZYUuqgVdFIq21TRLsdAOkV81En6OWWPh7tpLHGAtHlsDsCEZaHOxx03LLLHDuXnrnpmNDqqSZsyZ4lo4cLRblNVoCPIjpFl9nXELg0ulqXhVrsoYvYCevhYfyxE1bSzpShWjIEWcqgBmEt1DOvI3SrXrmpJ3iXwrhTSH1MlGLI0tmkzQBbNK/8RK5UDGWZmK8m3jGZjvOt3/12jTIFmv0VUQfIn+kS5hu2zGc7D65tRpFmObnYm5jztoo+gjRKLd+fSMOkHLNuDiEuhY1G0GRdkfWDD24MFKWYAKE7Q0ssg1I2g+6JzjOYUZoOM5xAG7BhQbwmWLcnEBUtyYNmuwPrAFMF22YUjBRQ7wkG3fn0gMt2RAEyEmo2MLughMC4NcQLYAd1TNdswV92NoSJhOOuITqnEtC/TYdScAQEHikzPdg9C35gfr8oZkrEaWSSWOSTUnziXLjUZqQkOAe0NpDOo4hLl/EwxyH69IrKYBB0jFcR+0fRyjTvEJ2opLmv8gx7mKlvtakVoEmH0Q/q1YDpcJwQR/b3jB6L7UdI5tdgh/eVl+ZII+sazQ8YkzlDJMUg7GoIPowxATiDXlTn18iIx/GeCPIczdNLMyU4CzdMtAy0ra8quKi4grTY42pGwNR/n+VEFeSeVOXWv6iEuJ1zOplcyfhsiYJSyyL1DS3SZ+zmNK+6pB3ZbnFQa0I6ARScOedKmHTze9lSwyG90/afs2rLMsGoJBOwwQSBjfrWv4Jpp7B5slGmDAelHpnFwoSMnG2Ygr2bVMLOm2VqstysxUIpS0kXYIrlifOLcuMSXm2+9VXCNYNAWDlBLZrzLBJYBqAuq0uVKhSA9N2GcGNzp9Qs5Q6MCpFQwNQRGPm9nXW4oyOHYF1YCjkCguuqGWn3ScbxM4DIOlLAIZaNkoWLLXmyVJp5gGlOVd8105tntqLrcbwAl2doCC8VMEzFTQbRl0H31MU2xB91TNdswYlA565hCzCcHnAHfdjaARbnflBsgUVG8EpuwYAfF5UgX243gObducGq3ZMAXd3ZrvAvgmmEYHKF2wBsgAJAij4vPJKpXA8R9dh+vzi2TcesU3GP90+iwDcswuXrZZUsHBANCQa5oDT6j5xEnfA38J/KMF261kyTp3WUxQAhBTkp896+e8bYqx7XfaHLkVlyv2kz8INAv8AG3L+EZ9Kxh5jTdSnf8QnOki6iyEIV3Nc2qQQoFCbnBJtNOZFJ2Qkq+slBhcKk0PUZEWPEZzPrGLkta8wCvIBio+gA9hGe741ebJ1mHtBw+XOrMEpZWmSpr4rn/mOSPUsOQG9OldluzyTNPehm6WpIAloiNQc2vRmNf3sxS9pJQlnQonhVwCwBOSACDXcZ6R07gOnUSrQMVO5J+8eZzG5fjHLPnbb/X0xfaHs13ch5rzjPly1LGXOlq5bkArLbac/dAPmI55N4dMl/wDE8NmTAuL5IqXlk7eEj9onLIuGK8o7J23mFNFNK4Pg5Aj/AHE5HEc37NH/AIojke7Ug5qrTLGBryKn9d4m74Wz4+k3sj9pfiWVqaKTs2yN7/cO/lHUZUxJi3Icc/L1/rHnntPKWs80yHmUPPE4IPXw79dzU5jcfY7r5rS7WdiquEAOaIR8PpEl1vqY6erUND/npElDja4dKeIf1im4FOZ5M0uSxSdNVSdwoY0Hn7xZS4CZLlruMqdxyiJq9PbkfCdx0MTBv7f0hU4eE+kRpXaJ7fAMAbRYS1BFTkxVJhgfMROrn3MKQt3IJAMOlAASBBpsPSGE3ERSpbEmhyIVN8IxiFTtobkbwCpXi3zCZjUNBgQeo5QuTtABVBAJGYRWEzNzDkB//9k=">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://png.pngtree.com/png-vector/20230830/ourlarge/pngtree-3d-businessman-asking-question-png-image_9187941.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''