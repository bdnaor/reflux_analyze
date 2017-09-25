const STATUS_INITIAL = 0, STATUS_SAVING = 1, STATUS_SUCCESS = 2, STATUS_FAILED = 3;

window.onload = function (){
    new Vue({
        el: '#app_control',
        data: {
            title: 'jjjdlkjf',
            models :{},
            mode: 'start',
            selected_model:'',
            work_header:'',
            uploadedFiles: [],
            uploadError: null,
            currentStatus: null,
            uploadFieldName: 'photos',
            predict_result : {}
        },
        computed: {
          isInitial: function() {
            return this.currentStatus === STATUS_INITIAL;
          },
          isSaving: function() {
            return this.currentStatus === STATUS_SAVING;
          },
          isSuccess: function() {
            return this.currentStatus === STATUS_SUCCESS;
          },
          isFailed: function() {
            return this.currentStatus === STATUS_FAILED;
          }
        },
        updated: function(){
            if(this.mode == 'model_details'){
                this.build_history_view();
                this.init_accordion();
            }
            if(this.mode == 'compere_models'){
                var myTextExtraction = function(node)
                {
                    // extract data from markup and return it
                    return node.childNodes[0].nodeValue;
                }
                $("#myTable").tablesorter({textExtraction: myTextExtraction});
                for (m in this.models){
                    var custom_id = m.replace(' ','')+"_row";
                    this.$refs[m][1].id = custom_id;

                    $('#'+custom_id).mouseenter(this.build_handler(custom_id, 'beige'));
                    $('#'+custom_id).mouseleave(this.build_handler(custom_id, 'white'));

                    // for(var i = 0; i<this.$refs[m][1].children.length; i++){
                    //     e = this.$refs[m][1].children[i];
                    //     e.id = custom_id+"_cell_"+i;
                    //     if (typeof e.id != 'string' || e.id == "" || e ==undefined)
                    //         return;
                    //     $('#'+e.id).mouseenter(function (event) {
                    //         for(var j = 0;j<$(this).parent()[0].children.length; j++) {
                    //             c = $(this).parent()[0].children[j];
                    //             c['background-color'] = 'beige';
                    //         }
                    //         // event.fromElement.parentElement.children('td, th').css('background-color','beige');
                    //     });
                    //
                    // }
                    // $('#'+custom_id).bind('hover', function(e) {
                    //     $(e.currentTarget).children('td, th').css('background-color','beige');
                    //     $(e.target).closest('tr').children('td,th').css('background-color','#000');
                    // })
                }

            }
        },
        mounted: function() {
            this.get_models();
            this.reset();
        },
        watch:{
            models:function(value){
                this.models = value;
            },
            predict_result: function(value){
                this.predict_result = value;
                // console.log(value)
            },
            mode: function (value) {
                if(value != 'model_predict'){
                    this.reset();
                }
            }
        },
        methods:{
            get_models: function () {
                var vm = this; // Keep reference to viewmodel object
                $.get('/get_models', function(data){
                    vm.models = JSON.parse(data);
                });
                setTimeout(function(){ vm.get_models() }, 700000);
            },
            showOptionList: function(e, divid, model){

                this.selected_model = model;

			    var left  = e.clientX  + "px";
			    var top  = e.clientY  + "px";

			    var div = document.getElementById(divid);

			    div.style.left = left;
			    div.style.top = top;

			    $("#"+divid).toggle();
			    $("#"+divid).mouseleave(function() {
                    $("#"+divid).css("display", "none");
                  });
			    $("#"+divid).click(function() {
                    $("#"+divid).css("display", "none");
                  });
			     e.preventDefault();
			    return false;
			},
            build_handler: function (id, myColor) {
              return function (event) {
                $('#'+id).children('td, th').css('background-color',myColor);
              }
            },
            show_details: function(){
                this.mode = 'model_details';
                this.work_header = 'Model Details';
            },
            create_model: function () {
                this.mode = 'create_model';
                this.work_header = 'Create New Model';
            },
            delete_model: function () {
                alert('delete!!!')
            },
            train_model: function () {
                this.mode = 'train_model';
                this.work_header = 'Train Model';
            },
            start_train: function (model) {
                vm = this;
                ep = this.$refs['epoch_train'].value;
                $.post('/start_train',{'model_name':model, 'epoch': ep},function(data, status){
                    if(status=="success"){
                        return;
                    }
                    else{
                        vm.openDialog('Fail start train');
                    }
                });
            },
            atTrainig: function (model) {
                var m = this.models[model];
                return m.total_train_epoch > m.done_train_epoch;
            },
            getTotalTrainEpoch: function (model) {
                var m = this.models[model];
                return m.total_train_epoch;
            },
            getDoneTrainEpoch: function (model) {
                var m = this.models[model];
                return m.done_train_epoch;
            },
            predict_model: function () {
                this.mode = 'model_predict';
                this.work_header = 'Predict Image';
                $('.plot-container').remove();
            },
            post_create: function () {
                var model_name = this.$refs['model_name'].value;
                var img_rows = this.$refs['img_rows'].value;
                var img_cols = this.$refs['img_cols'].value;
//                var epoch = this.$refs['epoch'].value;
                var kernel_size = this.$refs['kernel_size'].value;
                var pool_size = this.$refs['pool_size'].value;
                var activation_function = this.$refs['activation_function'].value;
                var dropout = this.$refs['dropout'].value;
                var nb_filters = this.$refs['nb_filters'].value;
                var train_ratio = this.$refs['train_ratio'].value;
                var split_cases = this.$refs['split_cases'].value;
                var batch_size = this.$refs['batch_size'].value;
                var sigma = this.$refs['sigma'].value;
                var theta = this.$refs['theta'].value;
                var lambd = this.$refs['lambd'].value;
                var gamma = this.$refs['gamma'].value;
                var psi = this.$refs['psi'].value;

                // TODO: add validation.
                var vm = this;
                $.post('/add_model',{
                    'model_name': model_name,
                    'img_rows': parseInt(img_rows),
                    'img_cols': parseInt(img_cols),
//                    'epoch': parseInt(epoch),
                    'kernel_size': parseInt(kernel_size),
                    'pool_size': parseInt(pool_size),
                    'activation_function': activation_function,
                    'dropout': parseFloat(dropout),
                    'nb_filters': parseInt(nb_filters),
                    'dropout': parseFloat(dropout),
                    'train_ratio': parseFloat(train_ratio),
                    'split_cases': split_cases,
                    'batch_size': parseInt(batch_size),
                    'sigma': parseFloat(sigma),
                    'theta': parseFloat(theta),
                    'lambd': parseFloat(lambd),
                    'gamma': parseFloat(gamma),
                    'psi': parseFloat(psi),

                },function(data,status){
                    if(status=="success"){
                        // models[model_name] = ;
                        vm.get_models();
                        vm.selected_model = vm.$refs['model_name'].value;
                        vm.show_details(vm.selected_model);
                        vm.$refs['model_name'].value = '';
                    }
                    else{
                        vm.openDialog(data.msg);
                    }
                })
            },
            reset: function() {
                // reset form to initial state
                this.currentStatus = STATUS_INITIAL;
                this.predict_result = {};
            },
            filesChange: function(fieldName, fileList) {
                // handle file changes
                var fd = new FormData();
                var vm = this; // Keep reference to viewmodel object

                var fileDict = {};
                if (!fileList.length) return;

                 // append the files to FormData
                for (x in Object.keys(fileList)){
                    fd.append(fileList[x].name, fileList[x]);
                    fileDict[fileList[x].name] = fileList[x];
                }
                fd.append('model_name', this.selected_model);

                function get_listener(i) {
                    load_lisinter = function (e) {
                        vm.$refs[i][0].src = e.target.result;
                    };
                    return load_lisinter
                }


                $.ajax({
                  url: '/predict_images',
                  data: fd,
                  processData: false,
                  contentType: false,
                  type: 'POST',
                  success: function(data){
                      for(i=0;i<Object.keys( data.predictions).length; i++){
                          var reader = new FileReader();
                          k = Object.keys( data.predictions)[i];
                          vm.predict_result[k] = {'result': data.predictions[k]};
                          reader.addEventListener('load',get_listener(k),false);
                          reader.readAsDataURL(fileDict[k]);
                      }
                      vm.currentStatus = STATUS_SUCCESS;
                      console.log(resultBuff);
                  },
                  error: function(response) {
					vm.openDialog(response.responseJSON.msg);
                  }
                });
            },
            getLatestInfo: function (item) {
                if(typeof item === 'string')
                    item = [item];
                var model_name = item.length > 1 ? item[1] : this.selected_model
                if(item[0] == 'train') {
                    var m = this.models[model_name];
                    var arr = m.con_mat_train[m.con_mat_train.length - 1];
                }
                else{
                    var m = this.models[model_name];
                    var arr = m.con_mat_val[m.con_mat_val.length - 1];
                }
                if (arr == undefined){
                    return {tn:0, fp:0, fn:0, tp:0};
                }
                return {tn:arr[0], fp:arr[1], fn:arr[2], tp:arr[3]};
            },
            total_population: function (item) {
                var m = this.getLatestInfo(item);
                return (m.tp + m.tn + m.fp + m.fn);
            },
            true_positive: function (item) {
                return this.getLatestInfo(item).tp
            },
            true_negative: function (item) {
                return this.getLatestInfo(item).tn
            },
            false_positive: function (item) {
                return this.getLatestInfo(item).fp
            },
            false_negative: function (item) {
                return this.getLatestInfo(item).fn
            },
            prevalence: function (item) {
                var tot_pos = this.true_negative(item) + this.false_positive(item);
                return ((tot_pos/this.total_population(item))*100).toFixed(2);
            },
            accuracy: function (item) {
                var tot_true = this.true_negative(item) + this.true_positive(item);
                return ((tot_true/this.total_population(item))*100).toFixed(2);
            },
            precision: function (item) {
                var tot_pos = this.true_positive(item) + this.false_positive(item);
                return ((this.true_positive(item)/tot_pos)*100).toFixed(2);
            },
            fdr: function (item) {
                var tot_pos = this.true_positive(item) + this.false_positive(item);
                return ((this.false_positive(item)/tot_pos)*100).toFixed(2);
            },
            false_omission_rate: function (item) {
                var tot_negative = this.false_negative(item) + this.true_negative(item);
                return ((this.false_negative(item)/tot_negative)*100).toFixed(2);
            },
            negative_predictive_value: function (item) {
                var tot_negative = this.false_negative(item) + this.true_negative(item);
                return ((this.true_negative(item)/tot_negative)*100).toFixed(2);
            },
            tpr: function(item){
                var tot_con_pos = this.true_positive(item) + this.false_negative(item);
                return ((this.true_positive(item)/tot_con_pos)*100).toFixed(2);
            },
            fnr: function(item){
                var tot_con_pos = this.true_positive(item) + this.false_negative(item);
                return ((this.false_negative(item)/tot_con_pos)*100).toFixed(2);
            },
            fpr: function (item) {
                var tot_con_neg = this.false_positive(item) + this.true_negative(item);
                return ((this.false_positive(item)/tot_con_neg)*100).toFixed(2);
            },
            tnr: function (item) {
                var tot_con_neg = this.false_positive(item) + this.true_negative(item);
                return ((this.true_negative(item)/tot_con_neg)*100).toFixed(2);
            },
            positive_likelihood_ratio: function (item) {
                return (this.tpr(item)/this.fpr(item)).toFixed(2);
            },
            negative_likelihood_ratio: function (item) {
                return (this.fnr(item)/this.tnr(item)).toFixed(2);
            },
            diagnostic_odds_ratio: function (item) {
                return (this.positive_likelihood_ratio(item)/this.negative_likelihood_ratio(item)).toFixed(2);
            },
            score: function (item) {
                // return (2*((this.tpr()*this.precision())/(this.tpr()+this.precision()))).toFixed(2);
                score = ((2/((1/this.tpr(item))+(1/this.precision(item))))/100).toFixed(2);
                if (score == "NaN")
                    return 0;
                return score;
            },
            getAccuracyDict: function (confusionList, name) {
                var x_acc = [];
                var y_acc = [];
                for(var i=0; i<confusionList.length; i++){
                    var arr = confusionList[i];
                    var mat = {tn:arr[0], fp:arr[1], fn:arr[2], tp:arr[3]};
                    x_acc.push(i);
                    var tot_true = mat.tn + mat.tp;
                    y_acc.push(((tot_true/(mat.tn+mat.fp+mat.fn+mat.tp))*100).toFixed(2));
                }
                 return {
                    x: x_acc,
                    y: y_acc,
                    name: name,
                    mode: "lines",
                    type: 'scatter',
                };
            },
            build_history_view: function () {

                var acc = this.getAccuracyDict(this.models[this.selected_model].con_mat_train, 'acc');

                var val_acc = this.getAccuracyDict(this.models[this.selected_model].con_mat_val, 'val_acc');

                var data = [acc, val_acc];

                Plotly.newPlot('history', data);

            },
            best_score: function(model, set){
                var scores = [];
                var arr = [];
                if(set=='test')
                    arr = this.models[model].con_mat_val;
                else
                    arr = this.models[model].con_mat_train;

                for (var i=0; i < arr.length; i++) {
                    var tn = arr[i][0];
                    var fp = arr[i][1];
                    var fn = arr[i][2];
                    var tp = arr[i][3]
                    scores.push(this.calculate_score(tn, fp, fn, tp));
                }
                var best = Math.max.apply(Math, scores);
                return best.toFixed(3);
            },
            calculate_score: function(tn, fp, fn, tp){
                if (((tp + fn) == 0) || ((tp+fp)==0))
                    return 0;
                _recall = tp / (tp + fn)
                _precision = tp / (tp+fp)
                if (_recall==0 || _precision==0)
                    return 0;
                return 2 / ((1 / _recall) + (1 / _precision))
            },
            init_accordion: function () {
                var acc = document.getElementsByClassName("accordion");
                var i;

                for (i = 0; i < acc.length; i++) {
                    acc[i].onclick = function(){
                        var cord = document.getElementsByClassName("accordion");
                        var panels = document.getElementsByClassName("panel");
                        var j;
                        for(j=0 ; j< cord.length; j++){
                            if(this != cord[j]) {
                                cord[j].classList.remove("active");
                                panels[j].style.display = "none";
                            }
                        }
                        /* Toggle between adding and removing the "active" class,
                        to highlight the button that controls the panel */
                        this.classList.toggle("active");

                        /* Toggle between hiding and showing the active panel */
                        var panel = this.nextElementSibling;
                        if (panel.style.display === "block") {
                            panel.style.display = "none";
                        } else {
                            panel.style.display = "block";
                        }
                    }
                }
            },
            openDialog: function (message) {
                this.$refs['dialog'].style.display = "block";
                this.$refs['dialog'].children[0].children[1].innerText = message;
                this.$refs['dialog'].children[0].children[2].onclick = this.closeDialog;
            },
            closeDialog: function () {
                this.$refs['dialog'].style.display = 'none';
            },
            compereModels: function(){
                this.mode = 'compere_models';
                this.work_header = 'Models Comparison';
                $("#myTable").tablesorter();
            },
            iterationTime: function(model){
                var diff = new Date(this.models[model].times_start_test[1])-new Date(this.models[model].times_finish[0]);
                var hh = Math.floor(diff / 1000 / 60 / 60);
                diff -= hh * 1000 * 60 * 60;
                var mm = Math.floor(diff / 1000 / 60);
                diff -= mm * 1000 * 60;
                var ss = Math.floor(diff / 1000);
                diff -= ss * 1000;
                if (hh > 0)
                    return hh+':'+mm+':'+ss
                return mm+':'+ss;
            }
        }
    });
};