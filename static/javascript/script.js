$(document).ready(function() {
  $('#customFile').on('change', function() {
      var fileName = $(this).val().split('\\').pop();
      $('#fileLabel').text(fileName);
  });
  
  $('#uploadForm').submit(function(event) {
    event.preventDefault();
    var formData = new FormData($(this)[0]);
 
    if ($('#customFile').val() === '') {
        alert("Choose A File First!");
        return; 
    }

    var file = $('#customFile')[0].files[0];
    // Periksa tipe file
    if (file.type !== 'application/vnd.ms-excel' && file.type !== 'text/csv') {
      alert("The file must be a CSV!");
      return; 
    }

    $('#circleloading').show();
    $('#loadinguploading').show();
    
    $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        async: true,
        cache: false,
        contentType: false,
        processData: false,
        success: function(response) {
          $('#message').text(response.message);
          $('#circleloading').hide();
          $('#loadinguploading').hide();
          window.location.href = '/konten';
        }
    });
  });
   $('#downloadGuideBtn').click(function() {
      $('#loadingdownloadguide').show();
    $.get('/download_guide', function(response) {
      window.location.href = '/download_guide';
      $('#loadingdownloadguide').hide();
      });
   });
    
   $('#viewDataFrameBtn').click(function() {
      $('#loadingviewdata').show();
    $.get('/view_dataframe', function(response) {
      $('#content').html(response.message);
      $('#loadingviewdata').hide();
      });
   });
    
   $('#infoDatasetBtn').click(function() {
      $('#loadinginfodata').show();
    $.get('/info_dataset', function(response) {
      $('#content').html(response.message);
      $('#loadinginfodata').hide();
      });
   });
    
    $('#checkNanBtn').click(function() {
      $('#loadingchecknan').show();
    $.get('/check_nan', function(response) {
      $('#content').html(response.message); 
      $('#loadingchecknan').hide();
      });
    });
    
   $('#removeNanBtn').click(function() {
      $('#loadingremovenan').show();
    $.get('/remove_nan', function(response) {
      $('#content').text(response.message);
      $('#loadingremovenan').hide();
      });
   });
    
   $('#changeDataTypeBtn').click(function() {
      $('#loadingchangedatatype').show();
    $.get('/change_data_type', function(response) {
      $('#content').text(response.message);
      $('#loadingchangedatatype').hide();
      });
   });

  /*
  $('#downloadCorrelationBtn').click(function() {
      $('#loadingdownloadcorrelationmatrix').show();
      $.get('/check_numeric', function(response) {
          if (response.message === "Data is already numeric.") {
              // Membuat URL download dengan parameter timestamp untuk menghindari caching
              var downloadUrl = '/download_correlation_matrix?t=' + new Date().getTime();

              // Mulai download
              window.location.href = downloadUrl;

              // Tambahkan sedikit waktu tunggu sebelum menghilangkan loading
              setTimeout(function() {
                  $('#loadingdownloadcorrelationmatrix').hide();
              }, 6300); // Waktu tunggu 2 detik, bisa disesuaikan
          } else {
              alert('The data has not been converted to numeric. Please change the data type first.');
              $('#loadingdownloadcorrelationmatrix').hide();
          }
      });
  });
  */
    
  $('#downloadCorrelationBtn').click(function() {
     $('#loadingdownloadcorrelationmatrix').show();
   $.get('/check_numeric', function(response) {
        if (response.message === "Data is already numeric.") {
            window.location.href = '/download_correlation_matrix';
            $('#loadingdownloadcorrelationmatrix').hide();
        } else {
            alert('The data has not been converted to numeric. Please change the data type first.');
            $('#loadingdownloadcorrelationmatrix').hide();
        }
    });
  });

    
   $('#removeColumnBtn').click(function() {
    var columnName = $('#columnNameToRemove').val(); 
      $('#loadingremovecolumn').show();
    $.get('/remove_column?column=' + columnName, function(response) {
      $('#content').text(response.message);
      $('#loadingremovecolumn').hide();
      });
   });
    
    $('#setTargetBtn').click(function() {
    var targetColumnName = $('#targetColumnName').val(); 
      $('#loadingsettarget').show();
    $.get('/set_target?column=' + targetColumnName, function(response) {
      $('#content').text(response.message);
      $('#loadingsettarget').hide();
      });
    });
    
    $('#splitDataBtn').click(function() {
    var testSize = $('#testSizeInput').val(); 
      $('#loadingsplitdata').show();
    $.get('/split_data?test_size=' + testSize, function(response) {
      $('#content').text(response.message);
      $('#loadingsplitdata').hide();
      });
    });
  
    $('#normalizeDataBtn').click(function() {
       $('#loadingnormalize').show();
     $.get('/normalize_data', function(response) {
       $('#content').text(response.message);
       $('#loadingnormalize').hide();
       });
    });

    /*
    $('#randomSearchBtn').click(function() {
        $('#loadingrandomsearch').show();
    
        $.get('/check_normalize', function(response) {
            if (response.message === "Data is already normalize.") {
                $.get('/random_search', function(response) {
                    var jsonString = JSON.stringify(response, null, 4);
    
                    var blob = new Blob([jsonString], { type: "application/json" });
    
                    var link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = 'random_search_results.txt';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
    
                    $('#loadingrandomsearch').hide(); selesai
                });
            } else {
                alert('The data has not been normalize. Please normalize the data first.');
                $('#loadingrandomsearch').hide(); 
            }
        });
    });
    */

    $('#gridSearchBtn').click(function() {
       $('#loadinggridsearch').show();

       $.get('/check_normalize', function(response) {
           if (response.message === "Data is already normalize.") {
               $.get('/grid_search', function(response) {
                   var jsonString = JSON.stringify(response, null, 4);

                   var blob = new Blob([jsonString], { type: "application/json" });

                   var link = document.createElement('a');
                   link.href = URL.createObjectURL(blob);
                   link.download = 'grid_search_results.txt';
                   document.body.appendChild(link);
                   link.click();
                   document.body.removeChild(link);

                   $('#loadinggridsearch').hide(); // Menyembunyikan loader setelah unduhan selesai
               });
           } else {
               alert('Data not scaled. Please normalize data first before performing Hyperparameter Tuning.');
               $('#loadinggridsearch').hide(); // Menyembunyikan loader jika data belum dinormalisasi
           }
       });
    });



   $('#tuningBtn').click(function() {
      $('#loadingtuning').show();
     $.get('/check_normalize', function(response) {
         if (response.message === "Data is already normalize.") {
             window.location.href = '/tuning';
             $('#loadingtuning').hide();
         } else {
             alert('Data not scaled. Please normalize data first before performing Hyperparameter Tuning.');
             $('#loadingtuning').hide();
         }
     });
   });
    
   $('#trainLRModelBtn').click(function() {
      $('#loadinglr').show();
     $.get('/train_logistic_regression', function(response) {
      $('#content').text(response.message);
      $('#loadinglr').hide();
      });
   });
    
   $('#trainDTModelBtn').click(function() {
      $('#loadingdt').show();
    $.get('/train_decision_tree', function(response) {
      $('#content').text(response.message);
      $('#loadingdt').hide();
      });
   });
    
    $('#trainXGBModelBtn').click(function() {
      $('#loadingxgb').show();
    $.get('/train_xgboost', function(response) {
      $('#content').text(response.message);
      $('#loadingxgb').hide();
      });
    });
    
   $('#viewModelResultBtn').click(function() {
      $('#loadingviewresult').show();
    $.get('/view_model_results', function(response) {
      $('#content').html(response.message);
      $('#loadingviewresult').hide();
      });
   });
    
    $('#viewShapBtn').click(function() {
      $('#loadingviewshap').show();
    $.get('/view_shap', function(response) {
      $('#content').html(response.message);
      $('#loadingviewshap').hide();
      });
    });  

    $('#downloadLrPredictedModelBtn').click(function() {
      $('#loadingdownloadlrpredictedmodel').show();
     $.get('/check_feature_testing_data_lr', function(response) {
      if (response.message === "Feature testing data are available.") {
        window.location.href = '/download_lr_predictions';
        $('#loadingdownloadlrpredictedmodel').hide();
      } else {
        alert('Feature testing data is not available. Please click view result button before download predicted LR CSV.');
        $('#loadingdownloadlrpredictedmodel').hide();
      }
     });
    });

  $('#downloadDtPredictedModelBtn').click(function() {
    $('#loadingdownloaddtpredictedmodel').show();
   $.get('/check_feature_testing_data_dt', function(response) {
    if (response.message === "Feature testing data are available.") {
      window.location.href = '/download_dt_predictions';
      $('#loadingdownloaddtpredictedmodel').hide();
    } else {
      alert('Feature testing data is not available. Please click view result button before download predicted DT CSV.');
      $('#loadingdownloaddtpredictedmodel').hide();
    }
   });
  });

  $('#downloadXgbPredictedModelBtn').click(function() {
    $('#loadingdownloadxgbpredictedmodel').show();
   $.get('/check_feature_testing_data_xgb', function(response) {
    if (response.message === "Feature testing data are available.") {
      window.location.href = '/download_xgb_predictions';
      $('#loadingdownloadxgbpredictedmodel').hide();
    } else {
      alert('Feature testing data is not available. Please click view result button before download predicted XGB CSV.');
      $('#loadingdownloadxgbpredictedmodel').hide();
    }
   });
  });
});                