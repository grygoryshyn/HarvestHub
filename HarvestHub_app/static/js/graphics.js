anychart.onDocumentReady(function () {
  
    // add data
    var data = [
      ["2003", 1, 0, 0],
      ["2004", 4, 0, 0],
      ["2005", 6, 0, 0],
      ["2006", 9, 1, 0],
      ["2007", 12, 2, 0],
      ["2008", 13, 5, 1],
      ["2009", 15, 6, 1],
      ["2010", 16, 9, 1],
      ["2011", 16, 10, 4],
      ["2012", 17, 11, 5],
      ["2013", 17, 13, 6],
      ["2014", 17, 14, 7],
      ["2015", 17, 14, 10],
      ["2016", 17, 14, 12],
      ["2017", 19, 16, 12],
      ["2018", 20, 17, 14],
      ["2019", 20, 19, 16],
      ["2020", 20, 20, 17],
      ["2021", 20, 20, 20],
      ["2022", 20, 22, 20]
    ];
    
    // create a data set
    var dataSet = anychart.data.set(data);
  
    // map the data for all series
    var firstSeriesData = dataSet.mapAs({x: 0, value: 1});
    var secondSeriesData = dataSet.mapAs({x: 0, value: 2});
    var thirdSeriesData = dataSet.mapAs({x: 0, value: 3});
  
    // create a stepped line chart
    var chart = anychart.stepLine();
  
    // create the series and name them
    var firstSeries = chart.stepLine(firstSeriesData);
    firstSeries.name("Roger Federer");
    var secondSeries = chart.stepLine(secondSeriesData);
    secondSeries.name("Rafael Nadal");
    var thirdSeries = chart.stepLine(thirdSeriesData);
    thirdSeries.name("Novak Djokovic");
  
    // add a legend and customize it
    chart.legend().enabled(true).fontSize(14).padding([10, 0, 10, 0]);
    
    // add a title and customize it
    chart
      .title()
      .enabled(true)
      .useHtml(true)
      .text(
        '<span style="color: #006331; font-size:20px;">Big Three&#39;s Grand Slam Title Race</span>' +
          '<br/><span style="font-size: 16px;">(Triumphs at Australian Open, French Open, Wimbledon, U.S. Open)</span>'
      );
    
    // name the axes
    chart.yAxis().title("Titles won");
    chart.xAxis().title("Year");
    
    // customize the series markers
    firstSeries.hovered().markers().type("circle").size(4);
    secondSeries.hovered().markers().type("circle").size(4);
    thirdSeries.hovered().markers().type("circle").size(4);
    
    // turn on crosshairs and remove the y hair
    chart.crosshair().enabled(true).yStroke(null).yLabel(false);
    
    // change the tooltip position
    chart.tooltip().positionMode("point");
    chart.tooltip().position("right").anchor("left-center").offsetX(5).offsetY(5);
    
    // customize the series stroke in the normal state
    firstSeries.normal().stroke("#7b60a2", 2.5);
    secondSeries.normal().stroke("#db7346", 2.5);
    thirdSeries.normal().stroke("#43a7dc", 2.5);
    
    // specify where to display the chart
    chart.container("container");
    
    // draw the resulting chart
    chart.draw();
    
  });